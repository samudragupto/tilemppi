"""Canonical NVIDIA cuTile (@ct.kernel) two-stage log-sum-exp softmax & ct.matmul update."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from tilemppi import cutile_compat as ct
from tilemppi.types import Float32Array


@ct.kernel
def cost_block_stats_kernel(
    costs_arr: Any,
    params_arr: Any,
    out_block_min_arr: Any,
    out_block_sum_arr: Any,
    TILE_K: ct.ConstInt,
) -> None:
    """Stage 1 @ct.kernel: compute per-tile minimum cost and local shifted sum-exp."""
    bid_k = ct.bid(0)

    c_tile = ct.load(costs_arr, index=(bid_k,), shape=(TILE_K,))
    params_tile = ct.load(params_arr, index=(0,), shape=(4,))
    lam_s = ct.extract(params_tile, index=(0,), shape=(1,))

    b_min = ct.reshape(ct.min(c_tile, axis=0, keepdims=True), (1,))
    b_min_b = ct.broadcast_to(b_min, (TILE_K,))
    lam_b = ct.broadcast_to(lam_s, (TILE_K,))

    shifted = ct.clip(
        (c_tile - b_min_b) / lam_b,
        ct.zeros((TILE_K,), dtype=ct.float32),
        ct.full((TILE_K,), 60.0, dtype=ct.float32),
    )
    exp_local = ct.exp(-shifted)
    b_sum = ct.reshape(ct.sum(exp_local, axis=0, keepdims=True), (1,))

    ct.store(out_block_min_arr, index=(bid_k,), tile=b_min)
    ct.store(out_block_sum_arr, index=(bid_k,), tile=b_sum)


@ct.kernel
def softmax_weighted_update_kernel(
    costs_arr: Any,
    perturbations_arr: Any,
    stats_arr: Any,
    out_weights_arr: Any,
    out_block_du_arr: Any,
    out_block_w2_arr: Any,
    TILE_K: ct.ConstInt,
    H: ct.ConstInt,
    D_U: ct.ConstInt,
) -> None:
    """Stage 2 @ct.kernel: compute normalized weights and ct.matmul weighted control update."""
    bid_k = ct.bid(0)

    c_tile = ct.load(costs_arr, index=(bid_k,), shape=(TILE_K,))
    stats_tile = ct.load(stats_arr, index=(0,), shape=(4,))

    g_min = ct.broadcast_to(ct.extract(stats_tile, index=(0,), shape=(1,)), (TILE_K,))
    g_sum = ct.broadcast_to(ct.extract(stats_tile, index=(1,), shape=(1,)), (TILE_K,))
    lam_b = ct.broadcast_to(ct.extract(stats_tile, index=(2,), shape=(1,)), (TILE_K,))

    shifted = ct.clip(
        (c_tile - g_min) / lam_b,
        ct.zeros((TILE_K,), dtype=ct.float32),
        ct.full((TILE_K,), 60.0, dtype=ct.float32),
    )
    w_tile = ct.exp(-shifted) / g_sum
    ct.store(out_weights_arr, index=(bid_k,), tile=w_tile)

    w2_sum = ct.reshape(ct.sum(w_tile * w_tile, axis=0, keepdims=True), (1,))
    ct.store(out_block_w2_arr, index=(bid_k,), tile=w2_sum)

    w_row = ct.reshape(w_tile, (1, TILE_K))
    for t in range(H):
        eps_3d = ct.load(perturbations_arr, index=(bid_k, t, 0), shape=(TILE_K, 1, D_U))
        eps_2d = ct.reshape(eps_3d, (TILE_K, D_U))
        du_row = ct.matmul(w_row, eps_2d)  # (1, D_U)
        ct.store(
            out_block_du_arr,
            index=(bid_k, t, 0),
            tile=ct.reshape(du_row, (1, 1, D_U)),
        )


def launch_softmax_update_cutile(
    costs: Float32Array,
    perturbations: Float32Array,
    nominal_controls: Float32Array,
    temperature: float,
    control_min: Float32Array,
    control_max: Float32Array,
    tile_k: int = 128,
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
    prefer_native: bool = False,
) -> tuple[Float32Array, Float32Array, float]:
    """Host launcher for two-stage cuTile softmax reduction & weighted control update."""
    c_arr = np.asarray(costs, dtype=np.float32).reshape(-1)
    num_rollouts = c_arr.shape[0]
    horizon, control_dim = nominal_controls.shape
    num_blocks = ct.cdiv(num_rollouts, tile_k)

    lam = float(max(temperature, 1e-4))
    params_arr = np.array([lam, 0.0, 0.0, 0.0], dtype=np.float32)
    block_min = np.zeros(num_blocks, dtype=np.float32)
    block_sum = np.zeros(num_blocks, dtype=np.float32)

    grid = (num_blocks, 1, 1)
    ct.launch(
        None,
        grid,
        cost_block_stats_kernel,
        (c_arr, params_arr, block_min, block_sum, tile_k),
        prefer_native=prefer_native,
    )

    # Combine block minimums and block sum-exps in float64/float32 exact log-sum-exp
    global_min = float(np.min(block_min))
    rescaled_sums = block_sum * np.exp(-np.clip((block_min - global_min) / lam, 0.0, 60.0))
    global_sum = float(max(float(np.sum(rescaled_sums)), 1e-12))

    stats_arr = np.array([global_min, global_sum, lam, 0.0], dtype=np.float32)
    out_weights = np.zeros(num_rollouts, dtype=np.float32)
    out_block_du = np.zeros((num_blocks, horizon, control_dim), dtype=np.float32)
    out_block_w2 = np.zeros(num_blocks, dtype=np.float32)

    ct.launch(
        None,
        grid,
        softmax_weighted_update_kernel,
        (
            c_arr,
            np.asarray(perturbations, dtype=np.float32),
            stats_arr,
            out_weights,
            out_block_du,
            out_block_w2,
            tile_k,
            horizon,
            control_dim,
        ),
        prefer_native=prefer_native,
    )

    delta_u = np.sum(out_block_du, axis=0).astype(np.float32)
    ess = float(1.0 / max(float(np.sum(out_block_w2)), 1e-12))

    u_new = np.clip(nominal_controls + delta_u, control_min, control_max).astype(np.float32)
    if savgol_window >= 3 and savgol_window <= horizon:
        u_smooth = savgol_filter(
            u_new,
            window_length=savgol_window,
            polyorder=savgol_polyorder,
            axis=0,
            mode="interp",
        )
        u_new = np.clip(u_smooth, control_min, control_max).astype(np.float32)

    return u_new, out_weights, ess

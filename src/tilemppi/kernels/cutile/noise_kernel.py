"""Canonical NVIDIA cuTile (@ct.kernel) stateless Philox PRNG + Ornstein-Uhlenbeck noise kernel."""

from __future__ import annotations

from typing import Any

import numpy as np

from tilemppi import cutile_compat as ct
from tilemppi.types import Float32Array, PrecisionMode


@ct.kernel
def philox_ou_noise_kernel(
    out_noise: Any,
    sigma_arr: Any,
    seed_arr: Any,
    params_arr: Any,
    TILE_K: ct.ConstInt,
    H: ct.ConstInt,
    D_U: ct.ConstInt,
) -> None:
    """Synthesize (TILE_K, H, D_U) Ornstein-Uhlenbeck colored Gaussian perturbations in tiles."""
    bid_k = ct.bid(0)

    sigma_tile = ct.load(sigma_arr, index=(0, 0), shape=(1, D_U))
    seed_tile = ct.load(seed_arr, index=(0,), shape=(1,))
    params_tile = ct.load(params_arr, index=(0,), shape=(4,))

    seed_u32 = ct.astype(seed_tile, ct.uint32)
    base_k = ct.full((TILE_K, 1), bid_k * TILE_K, dtype=ct.uint32)
    k_seq = ct.reshape(ct.arange(0, TILE_K, dtype=ct.uint32), (TILE_K, 1)) + base_k
    d_seq = ct.reshape(ct.arange(0, D_U, dtype=ct.uint32), (1, D_U))

    c_gold = ct.full((TILE_K, D_U), 0x9E3779B9, dtype=ct.uint32)
    c_m1 = ct.full((TILE_K, D_U), 0x85EBCA6B, dtype=ct.uint32)
    c_m2 = ct.full((TILE_K, D_U), 0xC2B2AE35, dtype=ct.uint32)
    c_m3 = ct.full((TILE_K, D_U), 0x27D4EB2D, dtype=ct.uint32)
    c_m4 = ct.full((TILE_K, D_U), 0x165667B1, dtype=ct.uint32)
    mask_24 = ct.full((TILE_K, D_U), 0x00FFFFFF, dtype=ct.uint32)

    s15 = ct.full((TILE_K, D_U), 15, dtype=ct.uint32)
    s13 = ct.full((TILE_K, D_U), 13, dtype=ct.uint32)
    s16 = ct.full((TILE_K, D_U), 16, dtype=ct.uint32)

    seed_b = ct.broadcast_to(ct.reshape(seed_u32, (1, 1)), (TILE_K, D_U))
    k_b = ct.broadcast_to(k_seq, (TILE_K, D_U))
    d_b = ct.broadcast_to(d_seq, (TILE_K, D_U))
    sig_b = ct.broadcast_to(sigma_tile, (TILE_K, D_U))

    rho = ct.extract(params_tile, index=(0,), shape=(1,))
    ou_scale = ct.extract(params_tile, index=(1,), shape=(1,))
    prec_flag = ct.extract(params_tile, index=(2,), shape=(1,))

    rho_b = ct.broadcast_to(ct.reshape(rho, (1, 1)), (TILE_K, D_U))
    scale_b = ct.broadcast_to(ct.reshape(ou_scale, (1, 1)), (TILE_K, D_U))

    inv_24 = ct.full((TILE_K, D_U), 1.0 / 16777217.0, dtype=ct.float32)
    two_pi_inv_24 = ct.full((TILE_K, D_U), 6.283185307179586 / 16777217.0, dtype=ct.float32)
    one_f = ct.ones((TILE_K, D_U), dtype=ct.float32)
    neg_two_f = ct.full((TILE_K, D_U), -2.0, dtype=ct.float32)

    prev_tile = ct.zeros((TILE_K, D_U), dtype=ct.float32)

    for t in range(H):
        t_u32 = ct.full((TILE_K, D_U), t, dtype=ct.uint32)
        t_mul = ct.full((TILE_K, D_U), 1664525, dtype=ct.uint32)
        d_mul = ct.full((TILE_K, D_U), 1013904223, dtype=ct.uint32)
        s_mul = ct.full((TILE_K, D_U), 2246822519, dtype=ct.uint32)

        with np.errstate(over="ignore"):
            c0 = k_b ^ (seed_b + c_gold)
            c1 = (t_u32 * t_mul) ^ (d_b * d_mul) ^ (seed_b * s_mul)

            h1 = (c0 ^ (c1 >> s15)) * c_m1
            h1 = (h1 ^ (h1 >> s13)) * c_m2
            u1_bits = h1 ^ (h1 >> s16)

            h2 = (c1 ^ (c0 >> s13) ^ c_m3) * c_m4
            h2 = (h2 ^ (h2 >> s15)) * c_m1
            u2_bits = h2 ^ (h2 >> s16)

        m1 = ct.astype(u1_bits & mask_24, ct.float32)
        m2 = ct.astype(u2_bits & mask_24, ct.float32)

        u1 = (m1 + one_f) * inv_24
        u2 = (m2 + one_f) * two_pi_inv_24

        z_tile = ct.sqrt(neg_two_f * ct.log(u1)) * ct.cos(u2)
        scaled = z_tile * sig_b

        if t == 0:
            curr = scaled
        else:
            curr = rho_b * prev_tile + scale_b * scaled

        if prec_flag[0] > 1.5:
            curr_q = ct.astype(curr, ct.float8_e4m3fn)
        elif prec_flag[0] > 0.5:
            curr_q = ct.astype(curr, ct.float16)
        else:
            curr_q = ct.astype(curr, ct.float32)

        prev_tile = curr_q
        ct.store(
            out_noise,
            index=(bid_k, t, 0),
            tile=ct.reshape(curr_q, (TILE_K, 1, D_U)),
        )


def launch_philox_ou_noise_cutile(
    num_rollouts: int,
    horizon: int,
    control_dim: int,
    sigma_vec: Float32Array,
    ou_rho: float,
    seed: int,
    tile_k: int = 128,
    precision: PrecisionMode = PrecisionMode.FP32,
    prefer_native: bool = False,
) -> Float32Array:
    """Host launcher for philox_ou_noise_kernel."""
    out_noise = np.zeros((num_rollouts, horizon, control_dim), dtype=np.float32)
    sigma_arr = np.asarray(sigma_vec, dtype=np.float32).reshape(1, control_dim)
    seed_arr = np.array([seed & 0xFFFFFFFF], dtype=np.uint32)

    prec_code = (
        2.0 if precision == PrecisionMode.FP8 else (1.0 if precision == PrecisionMode.FP16 else 0.0)
    )
    rho_f = np.float32(ou_rho)
    ou_scale_f = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - rho_f * rho_f))
    params_arr = np.array([rho_f, ou_scale_f, prec_code, 0.0], dtype=np.float32)

    grid = (ct.cdiv(num_rollouts, tile_k), 1, 1)
    ct.launch(
        None,
        grid,
        philox_ou_noise_kernel,
        (out_noise, sigma_arr, seed_arr, params_arr, tile_k, horizon, control_dim),
        prefer_native=prefer_native,
    )
    return out_noise

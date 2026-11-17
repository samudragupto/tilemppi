"""Canonical NVIDIA cuTile (@ct.kernel) batched quadruped forward kinematics & Jacobian kernel."""

from __future__ import annotations

from typing import Any

import numpy as np

from tilemppi import cutile_compat as ct
from tilemppi.dynamics.quadruped import HIP_OFFSETS_BODY, LINK_CALF, LINK_THIGH
from tilemppi.types import Float32Array, PrecisionMode


@ct.kernel
def batched_quadruped_fk_jacobian_kernel(
    states_arr: Any,
    forces_arr: Any,
    hip_offsets_arr: Any,
    params_arr: Any,
    out_feet_arr: Any,
    out_torques_arr: Any,
    TILE_K: ct.ConstInt,
) -> None:
    """Compute batched 4-leg 3-link FK and J^T * F torques using ct.matmul Tensor Core contractions."""
    bid_k = ct.bid(0)

    s_raw = ct.load(states_arr, index=(bid_k, 0), shape=(TILE_K, 32))
    f_raw = ct.load(forces_arr, index=(bid_k, 0, 0), shape=(TILE_K, 4, 3))
    hips_local = ct.load(hip_offsets_arr, index=(0, 0), shape=(4, 3))
    params_tile = ct.load(params_arr, index=(0,), shape=(4,))

    prec_code = ct.extract(params_tile, index=(2,), shape=(1,))
    if prec_code[0] > 1.5:
        s_tile = ct.astype(s_raw, ct.float8_e4m3fn)
        f_tile = ct.astype(f_raw, ct.float8_e4m3fn)
    elif prec_code[0] > 0.5:
        s_tile = ct.astype(s_raw, ct.float16)
        f_tile = ct.astype(f_raw, ct.float16)
    else:
        s_tile = ct.astype(s_raw, ct.float32)
        f_tile = ct.astype(f_raw, ct.float32)

    pos = ct.extract(s_tile, index=(0, 0), shape=(TILE_K, 3))
    qw = ct.extract(s_tile, index=(0, 3), shape=(TILE_K, 1))
    qx = ct.extract(s_tile, index=(0, 4), shape=(TILE_K, 1))
    qy = ct.extract(s_tile, index=(0, 5), shape=(TILE_K, 1))
    qz = ct.extract(s_tile, index=(0, 6), shape=(TILE_K, 1))

    one_f = ct.ones((TILE_K, 1), dtype=ct.float32)
    two_f = ct.full((TILE_K, 1), 2.0, dtype=ct.float32)

    r00 = one_f - two_f * (qy * qy + qz * qz)
    r01 = two_f * (qx * qy - qw * qz)
    r02 = two_f * (qx * qz + qw * qy)

    r10 = two_f * (qx * qy + qw * qz)
    r11 = one_f - two_f * (qx * qx + qz * qz)
    r12 = two_f * (qy * qz - qw * qx)

    r20 = two_f * (qx * qz - qw * qy)
    r21 = two_f * (qy * qz + qw * qx)
    r22 = one_f - two_f * (qx * qx + qy * qy)

    # Construct R^T of shape (TILE_K, 3, 3) so world_offset = body_offset @ R^T
    col0 = ct.reshape(ct.cat([r00, r01, r02], axis=-1), (TILE_K, 3, 1))
    col1 = ct.reshape(ct.cat([r10, r11, r12], axis=-1), (TILE_K, 3, 1))
    col2 = ct.reshape(ct.cat([r20, r21, r22], axis=-1), (TILE_K, 3, 1))
    rot_t = ct.cat([col0, col1, col2], axis=-1)

    l1_s = ct.extract(params_tile, index=(0,), shape=(1,))
    l2_s = ct.extract(params_tile, index=(1,), shape=(1,))
    l1 = ct.broadcast_to(ct.reshape(l1_s, (1, 1)), (TILE_K, 1))
    l2 = ct.broadcast_to(ct.reshape(l2_s, (1, 1)), (TILE_K, 1))
    zero_f = ct.zeros((TILE_K, 1), dtype=ct.float32)

    feet_body_legs: list[Any] = []
    torque_legs: list[Any] = []

    for leg in range(4):
        q_abd = ct.extract(s_tile, index=(0, 14 + 3 * leg), shape=(TILE_K, 1))
        q_hip = ct.extract(s_tile, index=(0, 15 + 3 * leg), shape=(TILE_K, 1))
        q_knee = ct.extract(s_tile, index=(0, 16 + 3 * leg), shape=(TILE_K, 1))
        q_sum = q_hip + q_knee

        c_abd = ct.cos(q_abd)
        s_abd = ct.sin(q_abd)
        c_hip = ct.cos(q_hip)
        s_hip = ct.sin(q_hip)
        c_sum = ct.cos(q_sum)
        s_sum = ct.sin(q_sum)

        leg_plane_z = l1 * c_hip + l2 * c_sum
        fx_rel = -(l1 * s_hip + l2 * s_sum)
        fy_rel = leg_plane_z * s_abd
        fz_rel = -leg_plane_z * c_abd

        hip_leg = ct.broadcast_to(
            ct.extract(hips_local, index=(leg, 0), shape=(1, 3)),
            (TILE_K, 3),
        )
        foot_leg_body = hip_leg + ct.cat([fx_rel, fy_rel, fz_rel], axis=-1)
        feet_body_legs.append(ct.reshape(foot_leg_body, (TILE_K, 1, 3)))

        # Analytical 3x3 Jacobian transpose J^T for this leg
        j00 = zero_f
        j01 = -leg_plane_z
        j02 = -l2 * c_sum

        j10 = leg_plane_z * c_abd
        j11 = fx_rel * s_abd
        j12 = -l2 * s_sum * s_abd

        j20 = leg_plane_z * s_abd
        j21 = -fx_rel * c_abd
        j22 = l2 * s_sum * c_abd

        # Rows of J^T are columns of J:
        jt_row0 = ct.reshape(ct.cat([j00, j10, j20], axis=-1), (TILE_K, 1, 3))
        jt_row1 = ct.reshape(ct.cat([j01, j11, j21], axis=-1), (TILE_K, 1, 3))
        jt_row2 = ct.reshape(ct.cat([j02, j12, j22], axis=-1), (TILE_K, 1, 3))
        jac_t_leg = ct.cat([jt_row0, jt_row1, jt_row2], axis=1)  # (TILE_K, 3, 3)

        f_leg = ct.extract(f_tile, index=(0, leg, 0), shape=(TILE_K, 1, 3))
        f_leg_col = ct.reshape(f_leg, (TILE_K, 3, 1))
        tau_leg_col = ct.matmul(jac_t_leg, f_leg_col)  # (TILE_K, 3, 1)
        torque_legs.append(ct.reshape(tau_leg_col, (TILE_K, 1, 3)))

    feet_body_all = ct.cat(feet_body_legs, axis=1)  # (TILE_K, 4, 3)
    pos_b = ct.broadcast_to(ct.reshape(pos, (TILE_K, 1, 3)), (TILE_K, 4, 3))
    feet_world_all = ct.mma(feet_body_all, rot_t, pos_b)
    torques_all = ct.cat(torque_legs, axis=1)

    ct.store(out_feet_arr, index=(bid_k, 0, 0), tile=feet_world_all)
    ct.store(out_torques_arr, index=(bid_k, 0, 0), tile=torques_all)


def launch_quadruped_fk_jacobian_cutile(
    states: Float32Array,
    foot_forces_body: Float32Array,
    tile_k: int = 128,
    precision: PrecisionMode = PrecisionMode.FP32,
    prefer_native: bool = False,
) -> tuple[Float32Array, Float32Array]:
    """Host launcher for batched_quadruped_fk_jacobian_kernel."""
    s_arr = np.asarray(states, dtype=np.float32).reshape(-1, 32)
    f_arr = np.asarray(foot_forces_body, dtype=np.float32).reshape(-1, 4, 3)
    n_batch = s_arr.shape[0]

    out_feet = np.zeros((n_batch, 4, 3), dtype=np.float32)
    out_torques = np.zeros((n_batch, 4, 3), dtype=np.float32)

    prec_code = (
        2.0 if precision == PrecisionMode.FP8 else (1.0 if precision == PrecisionMode.FP16 else 0.0)
    )
    params_arr = np.array([LINK_THIGH, LINK_CALF, prec_code, 0.0], dtype=np.float32)

    grid = (ct.cdiv(n_batch, tile_k), 1, 1)
    ct.launch(
        None,
        grid,
        batched_quadruped_fk_jacobian_kernel,
        (
            s_arr,
            f_arr,
            HIP_OFFSETS_BODY,
            params_arr,
            out_feet,
            out_torques,
            tile_k,
        ),
        prefer_native=prefer_native,
    )
    return out_feet, out_torques

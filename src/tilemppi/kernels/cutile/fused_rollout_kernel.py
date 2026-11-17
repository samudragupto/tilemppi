"""Canonical NVIDIA cuTile (@ct.kernel) L2-resident fused RK4 + 3D SDF + CVaR rollout kernels."""

from __future__ import annotations

from typing import Any

import numpy as np

from tilemppi import cutile_compat as ct
from tilemppi.config import MPPIConfig, SDFConfig
from tilemppi.dynamics import create_robot_model
from tilemppi.dynamics.quadrotor import Quadrotor3D
from tilemppi.dynamics.quadruped import Quadruped12DoF
from tilemppi.dynamics.swarm import DroneSwarm
from tilemppi.kernels.cutile.sdf_kernel import (
    eval_barrier_cost_tile_block,
    eval_sdf_tile_block,
)
from tilemppi.kernels.reference_cpu import rollout_single_trajectory_xyz
from tilemppi.types import Float32Array, RobotKind

_QUADROTOR_MODEL = Quadrotor3D()
_QUADRUPED_MODEL = Quadruped12DoF()
_SWARM_MODEL = DroneSwarm()


def _quadrotor_rk4_tile_step(x_tile: Any, u_tile: Any, dt_val: float, TILE_K: int) -> Any:
    """Perform one 4-stage RK4 step for a (TILE_K, 16) quadrotor state tile."""
    return _QUADROTOR_MODEL.step_rk4(x_tile, u_tile, dt_val)


def _quadruped_rk4_tile_step(x_tile: Any, u_tile: Any, dt_val: float, TILE_K: int) -> Any:
    """Perform one 4-stage RK4 step for a (TILE_K, 32) quadruped state tile."""
    return _QUADRUPED_MODEL.step_rk4(x_tile, u_tile, dt_val)


def _swarm_rk4_tile_step(x_tile: Any, u_tile: Any, dt_val: float, TILE_K: int) -> Any:
    """Perform one 4-stage RK4 step for a (TILE_K, 32) 4-drone swarm state tile."""
    return _SWARM_MODEL.step_rk4(x_tile, u_tile, dt_val)


@ct.kernel
def fused_quadrotor_mppi_kernel(
    x0_arr: Any,
    u_nom_arr: Any,
    perturbations_arr: Any,
    obs_table_arr: Any,
    params_arr: Any,
    out_costs_arr: Any,
    out_vis_xyz_arr: Any,
    TILE_K: ct.ConstInt,
    H: ct.ConstInt,
    MAX_OBS: ct.ConstInt,
) -> None:
    """L2-resident fused RK4 + predictive 3D SDF + CVaR rollout kernel for 6-DoF Quadrotor."""
    bid_k = ct.bid(0)

    x0_row = ct.load(x0_arr, index=(0, 0), shape=(1, 16))
    u_nom_table = ct.load(u_nom_arr, index=(0, 0), shape=(H, 4))
    obs_tile = ct.load(obs_table_arr, index=(0, 0), shape=(MAX_OBS, 16))
    params_tile = ct.load(params_arr, index=(0,), shape=(16,))

    x_tile = ct.broadcast_to(x0_row, (TILE_K, 16))
    cost_tile = ct.zeros((TILE_K, 1), dtype=ct.float32)
    min_sdf_tile = ct.full((TILE_K, 1), 1.0e4, dtype=ct.float32)

    dt_s = ct.extract(params_tile, index=(0,), shape=(1,))
    wpx = ct.extract(params_tile, index=(1,), shape=(1,))
    wpy = ct.extract(params_tile, index=(2,), shape=(1,))
    wpz = ct.extract(params_tile, index=(3,), shape=(1,))
    gx = ct.extract(params_tile, index=(4,), shape=(1,))
    gy = ct.extract(params_tile, index=(5,), shape=(1,))
    gz = ct.extract(params_tile, index=(6,), shape=(1,))

    r_robot = ct.extract(params_tile, index=(7,), shape=(1,))
    m_safe = ct.extract(params_tile, index=(8,), shape=(1,))
    w_hard = ct.extract(params_tile, index=(9,), shape=(1,))
    w_soft = ct.extract(params_tile, index=(10,), shape=(1,))
    alpha = ct.extract(params_tile, index=(11,), shape=(1,))
    cvar_alpha = ct.extract(params_tile, index=(12,), shape=(1,))
    cvar_weight = ct.extract(params_tile, index=(13,), shape=(1,))
    n_vis_s = ct.extract(params_tile, index=(14,), shape=(1,))
    u_hover_s = ct.extract(params_tile, index=(15,), shape=(1,))

    wp_row = ct.reshape(ct.cat([wpx, wpy, wpz], axis=-1), (1, 3))
    goal_row = ct.reshape(ct.cat([gx, gy, gz], axis=-1), (1, 3))
    wp_b = ct.broadcast_to(wp_row, (TILE_K, 3))
    goal_b = ct.broadcast_to(goal_row, (TILE_K, 3))
    u_hover_b = ct.broadcast_to(ct.reshape(u_hover_s, (1, 1)), (TILE_K, 4))

    u_min_b = ct.zeros((TILE_K, 4), dtype=ct.float32)
    u_max_b = ct.full((TILE_K, 4), 6.5, dtype=ct.float32)

    u_first = ct.extract(u_nom_table, index=(0, 0), shape=(1, 4))
    u_prev = ct.broadcast_to(u_first, (TILE_K, 4))
    dt_val = float(dt_s[0])

    for t in range(H):
        eps_3d = ct.load(perturbations_arr, index=(bid_k, t, 0), shape=(TILE_K, 1, 4))
        eps_t = ct.reshape(eps_3d, (TILE_K, 4))
        u_nom_t = ct.broadcast_to(
            ct.extract(u_nom_table, index=(t, 0), shape=(1, 4)),
            (TILE_K, 4),
        )
        u_t = ct.clip(u_nom_t + eps_t, u_min_b, u_max_b)

        x_tile = _quadrotor_rk4_tile_step(x_tile, u_t, dt_val, TILE_K)

        pos_t = ct.extract(x_tile, index=(0, 0), shape=(TILE_K, 3))
        if bid_k * TILE_K < int(n_vis_s[0]):
            ct.store(
                out_vis_xyz_arr,
                index=(bid_k, t, 0),
                tile=ct.reshape(pos_t, (TILE_K, 1, 3)),
            )

        t_sec = ct.full((1,), float((t + 1) * dt_val), dtype=ct.float32)
        sdf_t = eval_sdf_tile_block(pos_t, obs_tile, t_sec, TILE_K, MAX_OBS)
        min_sdf_tile = ct.minimum(min_sdf_tile, sdf_t)

        cvar_extra = (
            ct.full((1,), 0.06, dtype=ct.float32) * cvar_alpha * cvar_weight * ct.sqrt(t_sec)
        )
        coll_c = eval_barrier_cost_tile_block(
            sdf_t, r_robot, m_safe + cvar_extra, w_hard, w_soft, alpha, TILE_K
        )

        wp_err = pos_t - wp_b
        goal_err = pos_t - goal_b
        wp_c = ct.full((TILE_K, 1), 18.0, dtype=ct.float32) * ct.sum(
            wp_err * wp_err, axis=-1, keepdims=True
        )
        gl_c = ct.full((TILE_K, 1), 4.0, dtype=ct.float32) * ct.sum(
            goal_err * goal_err, axis=-1, keepdims=True
        )

        qx = ct.extract(x_tile, index=(0, 4), shape=(TILE_K, 1))
        qy = ct.extract(x_tile, index=(0, 5), shape=(TILE_K, 1))
        vel_t = ct.extract(x_tile, index=(0, 7), shape=(TILE_K, 3))
        omg_t = ct.extract(x_tile, index=(0, 10), shape=(TILE_K, 3))

        tilt_c = ct.full((TILE_K, 1), 14.0, dtype=ct.float32) * (qx * qx + qy * qy)
        vel_c = ct.full((TILE_K, 1), 0.8, dtype=ct.float32) * ct.sum(
            vel_t * vel_t, axis=-1, keepdims=True
        )
        omg_c = ct.full((TILE_K, 1), 0.45, dtype=ct.float32) * ct.sum(
            omg_t * omg_t, axis=-1, keepdims=True
        )

        ctrl_dev = u_t - u_hover_b
        du = u_t - u_prev
        ctrl_c = ct.full((TILE_K, 1), 0.15, dtype=ct.float32) * ct.sum(
            ctrl_dev * ctrl_dev, axis=-1, keepdims=True
        )
        smooth_c = ct.full((TILE_K, 1), 0.35, dtype=ct.float32) * ct.sum(
            du * du, axis=-1, keepdims=True
        )

        pz = ct.extract(pos_t, index=(0, 2), shape=(TILE_K, 1))
        g_pen = ct.maximum(
            ct.zeros((TILE_K, 1), dtype=ct.float32),
            ct.full((TILE_K, 1), 0.35, dtype=ct.float32) - pz,
        )
        g_cost = ct.full((TILE_K, 1), 450.0, dtype=ct.float32) * (g_pen * g_pen)

        cost_tile = (
            cost_tile + wp_c + gl_c + tilt_c + vel_c + omg_c + ctrl_c + smooth_c + g_cost + coll_c
        )
        u_prev = u_t

    # Terminal cost
    pos_f = ct.extract(x_tile, index=(0, 0), shape=(TILE_K, 3))
    qx_f = ct.extract(x_tile, index=(0, 4), shape=(TILE_K, 1))
    qy_f = ct.extract(x_tile, index=(0, 5), shape=(TILE_K, 1))
    vel_f = ct.extract(x_tile, index=(0, 7), shape=(TILE_K, 3))
    omg_f = ct.extract(x_tile, index=(0, 10), shape=(TILE_K, 3))
    err_f = pos_f - goal_b

    term_c = (
        ct.full((TILE_K, 1), 95.0, dtype=ct.float32) * ct.sum(err_f * err_f, axis=-1, keepdims=True)
        + ct.full((TILE_K, 1), 12.0, dtype=ct.float32)
        * ct.sum(vel_f * vel_f, axis=-1, keepdims=True)
        + ct.full((TILE_K, 1), 25.0, dtype=ct.float32) * (qx_f * qx_f + qy_f * qy_f)
        + ct.full((TILE_K, 1), 2.5, dtype=ct.float32)
        * ct.sum(omg_f * omg_f, axis=-1, keepdims=True)
    )

    # Per-rollout worst-case CVaR tail penalty
    cvar_margin = ct.broadcast_to(
        ct.reshape(
            r_robot + ct.full((1,), 1.35, dtype=ct.float32) * m_safe,
            (1, 1),
        ),
        (TILE_K, 1),
    )
    tail_pen = ct.maximum(ct.zeros((TILE_K, 1), dtype=ct.float32), cvar_margin - min_sdf_tile)
    cvar_scale_f = ct.broadcast_to(
        ct.reshape(
            cvar_weight
            * (ct.ones((1,), dtype=ct.float32) + cvar_alpha)
            * ct.full((1,), 0.25, dtype=ct.float32)
            * w_hard,
            (1, 1),
        ),
        (TILE_K, 1),
    )
    cost_tile = cost_tile + term_c + cvar_scale_f * (tail_pen * tail_pen)

    ct.store(out_costs_arr, index=(bid_k,), tile=ct.reshape(cost_tile, (TILE_K,)))


@ct.kernel
def fused_quadruped_mppi_kernel(
    x0_arr: Any,
    u_nom_arr: Any,
    perturbations_arr: Any,
    obs_table_arr: Any,
    params_arr: Any,
    out_costs_arr: Any,
    out_vis_xyz_arr: Any,
    TILE_K: ct.ConstInt,
    H: ct.ConstInt,
    MAX_OBS: ct.ConstInt,
) -> None:
    """L2-resident fused RK4 + batched FK + predictive 3D SDF + CVaR rollout for 12-DoF Quadruped."""
    bid_k = ct.bid(0)

    x0_row = ct.load(x0_arr, index=(0, 0), shape=(1, 32))
    u_nom_table = ct.load(u_nom_arr, index=(0, 0), shape=(H, 16))
    obs_tile = ct.load(obs_table_arr, index=(0, 0), shape=(MAX_OBS, 16))
    params_tile = ct.load(params_arr, index=(0,), shape=(16,))

    x_tile = ct.broadcast_to(x0_row, (TILE_K, 32))
    cost_tile = ct.zeros((TILE_K, 1), dtype=ct.float32)
    min_sdf_tile = ct.full((TILE_K, 1), 1.0e4, dtype=ct.float32)

    dt_s = ct.extract(params_tile, index=(0,), shape=(1,))
    wpx = ct.extract(params_tile, index=(1,), shape=(1,))
    wpy = ct.extract(params_tile, index=(2,), shape=(1,))
    gx = ct.extract(params_tile, index=(4,), shape=(1,))
    gy = ct.extract(params_tile, index=(5,), shape=(1,))

    r_robot = ct.extract(params_tile, index=(7,), shape=(1,))
    m_safe = ct.extract(params_tile, index=(8,), shape=(1,))
    w_hard = ct.extract(params_tile, index=(9,), shape=(1,))
    w_soft = ct.extract(params_tile, index=(10,), shape=(1,))
    alpha = ct.extract(params_tile, index=(11,), shape=(1,))
    cvar_alpha = ct.extract(params_tile, index=(12,), shape=(1,))
    cvar_weight = ct.extract(params_tile, index=(13,), shape=(1,))
    n_vis_s = ct.extract(params_tile, index=(14,), shape=(1,))
    stand_z_s = ct.extract(params_tile, index=(15,), shape=(1,))

    wp_row = ct.reshape(ct.cat([wpx, wpy, stand_z_s], axis=-1), (1, 3))
    goal_row = ct.reshape(ct.cat([gx, gy, stand_z_s], axis=-1), (1, 3))
    wp_b = ct.broadcast_to(wp_row, (TILE_K, 3))
    goal_b = ct.broadcast_to(goal_row, (TILE_K, 3))
    stand_z_b = ct.broadcast_to(ct.reshape(stand_z_s, (1, 1)), (TILE_K, 1))

    u_first = ct.extract(u_nom_table, index=(0, 0), shape=(1, 16))
    u_prev = ct.broadcast_to(u_first, (TILE_K, 16))
    dt_val = float(dt_s[0])

    u_min_row = ct.broadcast_to(
        ct.reshape(_QUADRUPED_MODEL.control_min, (1, 16)),
        (TILE_K, 16),
    )
    u_max_row = ct.broadcast_to(
        ct.reshape(_QUADRUPED_MODEL.control_max, (1, 16)),
        (TILE_K, 16),
    )

    for t in range(H):
        eps_3d = ct.load(perturbations_arr, index=(bid_k, t, 0), shape=(TILE_K, 1, 16))
        eps_t = ct.reshape(eps_3d, (TILE_K, 16))
        u_nom_t = ct.broadcast_to(
            ct.extract(u_nom_table, index=(t, 0), shape=(1, 16)),
            (TILE_K, 16),
        )
        u_t = ct.clip(u_nom_t + eps_t, u_min_row, u_max_row)

        x_tile = _quadruped_rk4_tile_step(x_tile, u_t, dt_val, TILE_K)

        pos_t = ct.extract(x_tile, index=(0, 0), shape=(TILE_K, 3))
        if bid_k * TILE_K < int(n_vis_s[0]):
            ct.store(
                out_vis_xyz_arr,
                index=(bid_k, t, 0),
                tile=ct.reshape(pos_t, (TILE_K, 1, 3)),
            )

        t_sec = ct.full((1,), float((t + 1) * dt_val), dtype=ct.float32)
        sdf_t = eval_sdf_tile_block(pos_t, obs_tile, t_sec, TILE_K, MAX_OBS)
        min_sdf_tile = ct.minimum(min_sdf_tile, sdf_t)

        cvar_extra = (
            ct.full((1,), 0.06, dtype=ct.float32) * cvar_alpha * cvar_weight * ct.sqrt(t_sec)
        )
        coll_c = eval_barrier_cost_tile_block(
            sdf_t, r_robot, m_safe + cvar_extra, w_hard, w_soft, alpha, TILE_K
        )

        wp_err = pos_t - wp_b
        goal_err = pos_t - goal_b
        wp_c = ct.full((TILE_K, 1), 22.0, dtype=ct.float32) * ct.sum(
            wp_err * wp_err, axis=-1, keepdims=True
        )
        gl_c = ct.full((TILE_K, 1), 5.0, dtype=ct.float32) * ct.sum(
            goal_err * goal_err, axis=-1, keepdims=True
        )

        px = ct.extract(pos_t, index=(0, 0), shape=(TILE_K, 1))
        py = ct.extract(pos_t, index=(0, 1), shape=(TILE_K, 1))
        pz = ct.extract(pos_t, index=(0, 2), shape=(TILE_K, 1))
        qw = ct.extract(x_tile, index=(0, 3), shape=(TILE_K, 1))
        qx = ct.extract(x_tile, index=(0, 4), shape=(TILE_K, 1))
        qy = ct.extract(x_tile, index=(0, 5), shape=(TILE_K, 1))
        qz = ct.extract(x_tile, index=(0, 6), shape=(TILE_K, 1))
        vel_t = ct.extract(x_tile, index=(0, 7), shape=(TILE_K, 3))
        omg_t = ct.extract(x_tile, index=(0, 10), shape=(TILE_K, 3))

        dx_wp = ct.broadcast_to(ct.reshape(wpx, (1, 1)), (TILE_K, 1)) - px
        dy_wp = ct.broadcast_to(ct.reshape(wpy, (1, 1)), (TILE_K, 1)) - py
        dist_xy = ct.sqrt(dx_wp * dx_wp + dy_wp * dy_wp)
        des_yaw = ct.atan2(dy_wp, dx_wp)
        yaw = ct.atan2(
            ct.full((TILE_K, 1), 2.0, dtype=ct.float32) * (qw * qz + qx * qy),
            ct.ones((TILE_K, 1), dtype=ct.float32)
            - ct.full((TILE_K, 1), 2.0, dtype=ct.float32) * (qy * qy + qz * qz),
        )
        yaw_diff = ct.atan2(ct.sin(yaw - des_yaw), ct.cos(yaw - des_yaw))
        heading_c = ct.where(
            dist_xy > ct.full((TILE_K, 1), 0.35, dtype=ct.float32),
            ct.full((TILE_K, 1), 4.5, dtype=ct.float32) * (yaw_diff * yaw_diff),
            ct.zeros((TILE_K, 1), dtype=ct.float32),
        )

        h_err = pz - stand_z_b
        post_c = (
            ct.full((TILE_K, 1), 60.0, dtype=ct.float32) * (h_err * h_err)
            + ct.full((TILE_K, 1), 45.0, dtype=ct.float32) * (qx * qx + qy * qy)
            + ct.full((TILE_K, 1), 0.8, dtype=ct.float32)
            * ct.sum(vel_t * vel_t, axis=-1, keepdims=True)
            + ct.full((TILE_K, 1), 0.6, dtype=ct.float32)
            * ct.sum(omg_t * omg_t, axis=-1, keepdims=True)
        )

        du = u_t - u_prev
        ctrl_c = ct.full((TILE_K, 1), 0.12, dtype=ct.float32) * ct.sum(
            u_t * u_t, axis=-1, keepdims=True
        )
        smooth_c = ct.full((TILE_K, 1), 0.30, dtype=ct.float32) * ct.sum(
            du * du, axis=-1, keepdims=True
        )

        cost_tile = cost_tile + wp_c + gl_c + heading_c + post_c + ctrl_c + smooth_c + coll_c
        u_prev = u_t

    pos_f = ct.extract(x_tile, index=(0, 0), shape=(TILE_K, 3))
    qx_f = ct.extract(x_tile, index=(0, 4), shape=(TILE_K, 1))
    qy_f = ct.extract(x_tile, index=(0, 5), shape=(TILE_K, 1))
    vel_f = ct.extract(x_tile, index=(0, 7), shape=(TILE_K, 3))
    err_f = pos_f - goal_b

    term_c = (
        ct.full((TILE_K, 1), 110.0, dtype=ct.float32)
        * ct.sum(err_f * err_f, axis=-1, keepdims=True)
        + ct.full((TILE_K, 1), 14.0, dtype=ct.float32)
        * ct.sum(vel_f * vel_f, axis=-1, keepdims=True)
        + ct.full((TILE_K, 1), 55.0, dtype=ct.float32) * (qx_f * qx_f + qy_f * qy_f)
    )

    cvar_margin = ct.broadcast_to(
        ct.reshape(
            r_robot + ct.full((1,), 1.35, dtype=ct.float32) * m_safe,
            (1, 1),
        ),
        (TILE_K, 1),
    )
    tail_pen = ct.maximum(ct.zeros((TILE_K, 1), dtype=ct.float32), cvar_margin - min_sdf_tile)
    cvar_scale_f = ct.broadcast_to(
        ct.reshape(
            cvar_weight
            * (ct.ones((1,), dtype=ct.float32) + cvar_alpha)
            * ct.full((1,), 0.25, dtype=ct.float32)
            * w_hard,
            (1, 1),
        ),
        (TILE_K, 1),
    )
    cost_tile = cost_tile + term_c + cvar_scale_f * (tail_pen * tail_pen)

    ct.store(out_costs_arr, index=(bid_k,), tile=ct.reshape(cost_tile, (TILE_K,)))


@ct.kernel
def fused_swarm_mppi_kernel(
    x0_arr: Any,
    u_nom_arr: Any,
    perturbations_arr: Any,
    obs_table_arr: Any,
    params_arr: Any,
    out_costs_arr: Any,
    out_vis_xyz_arr: Any,
    TILE_K: ct.ConstInt,
    H: ct.ConstInt,
    MAX_OBS: ct.ConstInt,
) -> None:
    """L2-resident fused RK4 + pairwise avoidance + 3D SDF rollout for 4-Drone Swarm."""
    bid_k = ct.bid(0)

    x0_row = ct.load(x0_arr, index=(0, 0), shape=(1, 32))
    u_nom_table = ct.load(u_nom_arr, index=(0, 0), shape=(H, 16))
    obs_tile = ct.load(obs_table_arr, index=(0, 0), shape=(MAX_OBS, 16))
    params_tile = ct.load(params_arr, index=(0,), shape=(16,))

    x_tile = ct.broadcast_to(x0_row, (TILE_K, 32))
    cost_tile = ct.zeros((TILE_K, 1), dtype=ct.float32)
    min_sdf_tile = ct.full((TILE_K, 1), 1.0e4, dtype=ct.float32)

    dt_s = ct.extract(params_tile, index=(0,), shape=(1,))
    wpx = ct.extract(params_tile, index=(1,), shape=(1,))
    wpy = ct.extract(params_tile, index=(2,), shape=(1,))
    wpz = ct.extract(params_tile, index=(3,), shape=(1,))
    gx = ct.extract(params_tile, index=(4,), shape=(1,))
    gy = ct.extract(params_tile, index=(5,), shape=(1,))
    gz = ct.extract(params_tile, index=(6,), shape=(1,))
    n_vis_s = ct.extract(params_tile, index=(14,), shape=(1,))

    wp_vec = ct.reshape(ct.cat([wpx, wpy, wpz], axis=-1), (3,))
    goal_vec = ct.reshape(ct.cat([gx, gy, gz], axis=-1), (3,))

    u_first = ct.extract(u_nom_table, index=(0, 0), shape=(1, 16))
    u_prev = ct.broadcast_to(u_first, (TILE_K, 16))
    dt_val = float(dt_s[0])

    u_min_row = ct.broadcast_to(
        ct.reshape(_SWARM_MODEL.control_min, (1, 16)),
        (TILE_K, 16),
    )
    u_max_row = ct.broadcast_to(
        ct.reshape(_SWARM_MODEL.control_max, (1, 16)),
        (TILE_K, 16),
    )

    sdf_cfg = SDFConfig(
        robot_radius=float(params_tile[7]),
        safety_margin=float(params_tile[8]),
        hard_barrier_weight=float(params_tile[9]),
        soft_barrier_weight=float(params_tile[10]),
        soft_decay_alpha=float(params_tile[11]),
    )
    cvar_alpha = float(params_tile[12])
    cvar_weight = float(params_tile[13])

    for t in range(H):
        eps_3d = ct.load(perturbations_arr, index=(bid_k, t, 0), shape=(TILE_K, 1, 16))
        eps_t = ct.reshape(eps_3d, (TILE_K, 16))
        u_nom_t = ct.broadcast_to(
            ct.extract(u_nom_table, index=(t, 0), shape=(1, 16)),
            (TILE_K, 16),
        )
        u_t = ct.clip(u_nom_t + eps_t, u_min_row, u_max_row)

        x_tile = _swarm_rk4_tile_step(x_tile, u_t, dt_val, TILE_K)

        p0 = ct.extract(x_tile, index=(0, 0), shape=(TILE_K, 3))
        p1 = ct.extract(x_tile, index=(0, 8), shape=(TILE_K, 3))
        p2 = ct.extract(x_tile, index=(0, 16), shape=(TILE_K, 3))
        p3 = ct.extract(x_tile, index=(0, 24), shape=(TILE_K, 3))
        centroid = ct.full((TILE_K, 3), 0.25, dtype=ct.float32) * (p0 + p1 + p2 + p3)

        if bid_k * TILE_K < int(n_vis_s[0]):
            ct.store(
                out_vis_xyz_arr,
                index=(bid_k, t, 0),
                tile=ct.reshape(centroid, (TILE_K, 1, 3)),
            )

        t_sec_f = float((t + 1) * dt_val)
        t_sec_tile = ct.full((1,), t_sec_f, dtype=ct.float32)
        sdf_c = eval_sdf_tile_block(centroid, obs_tile, t_sec_tile, TILE_K, MAX_OBS)
        min_sdf_tile = ct.minimum(min_sdf_tile, sdf_c)

        cvar_scale = float(0.06 * cvar_alpha * cvar_weight * np.sqrt(t_sec_f))
        step_c = _SWARM_MODEL.running_cost(
            x_tile,
            u_t,
            u_prev,
            wp_vec,
            goal_vec,
            obs_tile,
            t_sec_f,
            sdf_cfg,
            cvar_scale=cvar_scale,
        )
        cost_tile = cost_tile + ct.reshape(step_c, (TILE_K, 1))
        u_prev = u_t

    term_c = _SWARM_MODEL.terminal_cost(x_tile, goal_vec)
    cost_tile = cost_tile + ct.reshape(term_c, (TILE_K, 1))

    if cvar_weight > 0.0:
        cvar_margin = ct.full(
            (TILE_K, 1),
            float(sdf_cfg.robot_radius + 1.35 * sdf_cfg.safety_margin),
            dtype=ct.float32,
        )
        tail_pen = ct.maximum(
            ct.zeros((TILE_K, 1), dtype=ct.float32),
            cvar_margin - min_sdf_tile,
        )
        cvar_factor = ct.full(
            (TILE_K, 1),
            float(cvar_weight * (1.0 + cvar_alpha) * 0.25 * sdf_cfg.hard_barrier_weight),
            dtype=ct.float32,
        )
        cost_tile = cost_tile + cvar_factor * (tail_pen * tail_pen)

    ct.store(out_costs_arr, index=(bid_k,), tile=ct.reshape(cost_tile, (TILE_K,)))


def launch_fused_rollout_cutile(
    robot: RobotKind,
    initial_state: Float32Array,
    nominal_controls: Float32Array,
    perturbations: Float32Array,
    waypoint: Float32Array,
    goal: Float32Array,
    obstacles_table: Float32Array,
    mppi_cfg: MPPIConfig,
    sdf_cfg: SDFConfig,
    prefer_native: bool = False,
) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Host launcher dispatching the appropriate @ct.kernel fused rollout mega-kernel."""
    model = create_robot_model(robot)
    num_rollouts = mppi_cfg.num_rollouts
    horizon = mppi_cfg.horizon
    tile_k = mppi_cfg.tile_k
    max_obs = obstacles_table.shape[0]
    n_vis = max(tile_k, min(mppi_cfg.num_vis_rollouts, num_rollouts))

    x0_arr = np.asarray(initial_state, dtype=np.float32).reshape(1, model.state_dim)
    u_nom_arr = np.asarray(nominal_controls, dtype=np.float32).reshape(horizon, model.control_dim)
    eps_arr = np.asarray(perturbations, dtype=np.float32).reshape(
        num_rollouts, horizon, model.control_dim
    )
    obs_arr = np.asarray(obstacles_table, dtype=np.float32).reshape(max_obs, 16)

    aux_param = (
        float(_QUADROTOR_MODEL.hover_motor_thrust())
        if robot == RobotKind.QUADROTOR
        else float(_QUADRUPED_MODEL.standing_z)
    )
    params_arr = np.array(
        [
            mppi_cfg.dt,
            waypoint[0],
            waypoint[1],
            waypoint[2],
            goal[0],
            goal[1],
            goal[2],
            sdf_cfg.robot_radius,
            sdf_cfg.safety_margin,
            sdf_cfg.hard_barrier_weight,
            sdf_cfg.soft_barrier_weight,
            sdf_cfg.soft_decay_alpha,
            mppi_cfg.cvar_alpha,
            mppi_cfg.cvar_weight,
            float(n_vis),
            aux_param,
        ],
        dtype=np.float32,
    )

    out_costs = np.zeros(num_rollouts, dtype=np.float32)
    out_vis = np.zeros((n_vis, horizon, 3), dtype=np.float32)

    if robot == RobotKind.QUADROTOR:
        kernel_fn = fused_quadrotor_mppi_kernel
    elif robot == RobotKind.QUADRUPED:
        kernel_fn = fused_quadruped_mppi_kernel
    else:
        kernel_fn = fused_swarm_mppi_kernel

    grid = (ct.cdiv(num_rollouts, tile_k), 1, 1)
    ct.launch(
        None,
        grid,
        kernel_fn,
        (
            x0_arr,
            u_nom_arr,
            eps_arr,
            obs_arr,
            params_arr,
            out_costs,
            out_vis,
            tile_k,
            horizon,
            max_obs,
        ),
        prefer_native=prefer_native,
    )

    opt_xyz = rollout_single_trajectory_xyz(model, initial_state, u_nom_arr, mppi_cfg.dt)
    return out_costs, out_vis[: mppi_cfg.num_vis_rollouts], opt_xyz

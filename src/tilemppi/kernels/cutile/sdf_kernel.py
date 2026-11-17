"""Canonical NVIDIA cuTile (@ct.kernel) 3D Signed Distance Function & barrier collision kernel."""

from __future__ import annotations

from typing import Any

import numpy as np

from tilemppi import cutile_compat as ct
from tilemppi.config import SDFConfig
from tilemppi.types import Float32Array


def eval_sdf_tile_block(
    pts_tile: Any,
    obs_tile: Any,
    t_val: Any,
    TILE_K: int,
    MAX_OBS: int,
) -> Any:
    """Evaluate minimum 3D SDF distance tile (TILE_K, 1) across MAX_OBS dynamic obstacles."""
    px = ct.extract(pts_tile, index=(0, 0), shape=(TILE_K, 1))
    py = ct.extract(pts_tile, index=(0, 1), shape=(TILE_K, 1))
    pz = ct.extract(pts_tile, index=(0, 2), shape=(TILE_K, 1))

    min_d = ct.full((TILE_K, 1), 1.0e4, dtype=ct.float32)
    zero_f = ct.zeros((TILE_K, 1), dtype=ct.float32)
    one_f = ct.ones((TILE_K, 1), dtype=ct.float32)
    two_f = ct.full((TILE_K, 1), 2.0, dtype=ct.float32)
    half_f = ct.full((TILE_K, 1), 0.5, dtype=ct.float32)
    far_f = ct.full((TILE_K, 1), 1.0e4, dtype=ct.float32)

    t_b = ct.broadcast_to(ct.reshape(t_val, (1, 1)), (TILE_K, 1))
    t2_half = half_f * (t_b * t_b)

    for m in range(MAX_OBS):
        row = ct.extract(obs_tile, index=(m, 0), shape=(1, 16))
        kind_s = ct.extract(row, index=(0, 0), shape=(1, 1))
        kind_b = ct.broadcast_to(kind_s, (TILE_K, 1))

        ox = ct.broadcast_to(ct.extract(row, index=(0, 1), shape=(1, 1)), (TILE_K, 1))
        oy = ct.broadcast_to(ct.extract(row, index=(0, 2), shape=(1, 1)), (TILE_K, 1))
        oz = ct.broadcast_to(ct.extract(row, index=(0, 3), shape=(1, 1)), (TILE_K, 1))

        vx = ct.broadcast_to(ct.extract(row, index=(0, 4), shape=(1, 1)), (TILE_K, 1))
        vy = ct.broadcast_to(ct.extract(row, index=(0, 5), shape=(1, 1)), (TILE_K, 1))
        vz = ct.broadcast_to(ct.extract(row, index=(0, 6), shape=(1, 1)), (TILE_K, 1))

        ax = ct.broadcast_to(ct.extract(row, index=(0, 7), shape=(1, 1)), (TILE_K, 1))
        ay = ct.broadcast_to(ct.extract(row, index=(0, 8), shape=(1, 1)), (TILE_K, 1))
        az = ct.broadcast_to(ct.extract(row, index=(0, 9), shape=(1, 1)), (TILE_K, 1))

        p0 = ct.broadcast_to(ct.extract(row, index=(0, 10), shape=(1, 1)), (TILE_K, 1))
        p1 = ct.broadcast_to(ct.extract(row, index=(0, 11), shape=(1, 1)), (TILE_K, 1))
        p2 = ct.broadcast_to(ct.extract(row, index=(0, 12), shape=(1, 1)), (TILE_K, 1))

        qx = ct.broadcast_to(ct.extract(row, index=(0, 13), shape=(1, 1)), (TILE_K, 1))
        qy = ct.broadcast_to(ct.extract(row, index=(0, 14), shape=(1, 1)), (TILE_K, 1))
        qz = ct.broadcast_to(ct.extract(row, index=(0, 15), shape=(1, 1)), (TILE_K, 1))
        qw = ct.sqrt(ct.maximum(zero_f, one_f - (qx * qx + qy * qy + qz * qz)))

        cx = ox + vx * t_b + ax * t2_half
        cy = oy + vy * t_b + ay * t2_half
        cz = oz + vz * t_b + az * t2_half

        dx = px - cx
        dy = py - cy
        dz = pz - cz

        # Inverse quaternion rotation into obstacle local frame
        ux = -qx
        uy = -qy
        uz = -qz
        tx = two_f * (uy * dz - uz * dy)
        ty = two_f * (uz * dx - ux * dz)
        tz = two_f * (ux * dy - uy * dx)

        lx = dx + qw * tx + (uy * tz - uz * ty)
        ly = dy + qw * ty + (uz * tx - ux * tz)
        lz = dz + qw * tz + (ux * ty - uy * tx)

        # 0: Sphere
        d_sphere = ct.sqrt(lx * lx + ly * ly + lz * lz) - p0

        # 1: Capsule
        cz_clamp = ct.clip(lz, -p1, p1)
        dz_cap = lz - cz_clamp
        d_capsule = ct.sqrt(lx * lx + ly * ly + dz_cap * dz_cap) - p0

        # 2: OrientedBox
        bx = ct.abs(lx) - p0
        by = ct.abs(ly) - p1
        bz = ct.abs(lz) - p2
        ox_b = ct.maximum(bx, zero_f)
        oy_b = ct.maximum(by, zero_f)
        oz_b = ct.maximum(bz, zero_f)
        d_box = ct.sqrt(ox_b * ox_b + oy_b * oy_b + oz_b * oz_b) + ct.minimum(
            ct.maximum(bx, ct.maximum(by, bz)), zero_f
        )

        # 3: TorusGate (ring in YZ plane)
        rho_yz = ct.sqrt(ly * ly + lz * lz) - p0
        d_torus = ct.sqrt(lx * lx + rho_yz * rho_yz) - p1

        # 4: GroundPlane
        d_ground = pz - (cz + p0)

        d_prim = ct.where(
            kind_b < ct.full((TILE_K, 1), -0.5, dtype=ct.float32),
            far_f,
            ct.where(
                kind_b < ct.full((TILE_K, 1), 0.5, dtype=ct.float32),
                d_sphere,
                ct.where(
                    kind_b < ct.full((TILE_K, 1), 1.5, dtype=ct.float32),
                    d_capsule,
                    ct.where(
                        kind_b < ct.full((TILE_K, 1), 2.5, dtype=ct.float32),
                        d_box,
                        ct.where(
                            kind_b < ct.full((TILE_K, 1), 3.5, dtype=ct.float32),
                            d_torus,
                            d_ground,
                        ),
                    ),
                ),
            ),
        )
        min_d = ct.minimum(min_d, d_prim)

    return min_d


def eval_barrier_cost_tile_block(
    sdf_tile: Any,
    r_robot: Any,
    m_safe: Any,
    w_hard: Any,
    w_soft: Any,
    alpha: Any,
    TILE_K: int,
) -> Any:
    """Compute differentiable collision barrier cost tile (TILE_K, 1) from SDF distances."""
    zero_f = ct.zeros((TILE_K, 1), dtype=ct.float32)
    r_b = ct.broadcast_to(ct.reshape(r_robot, (1, 1)), (TILE_K, 1))
    m_b = ct.broadcast_to(ct.reshape(m_safe, (1, 1)), (TILE_K, 1))
    wh_b = ct.broadcast_to(ct.reshape(w_hard, (1, 1)), (TILE_K, 1))
    ws_b = ct.broadcast_to(ct.reshape(w_soft, (1, 1)), (TILE_K, 1))
    al_b = ct.broadcast_to(ct.reshape(alpha, (1, 1)), (TILE_K, 1))

    pen = ct.maximum(zero_f, (r_b + m_b) - sdf_tile)
    hard_c = wh_b * (pen * pen)
    exp_arg = ct.clip(
        -al_b * (sdf_tile - r_b),
        ct.full((TILE_K, 1), -30.0, dtype=ct.float32),
        ct.full((TILE_K, 1), 15.0, dtype=ct.float32),
    )
    soft_c = ws_b * ct.exp(exp_arg)
    return hard_c + soft_c


@ct.kernel
def tiled_sdf_collision_kernel(
    points_arr: Any,
    obs_table_arr: Any,
    params_arr: Any,
    out_sdf_arr: Any,
    out_barrier_arr: Any,
    out_normals_arr: Any,
    TILE_K: ct.ConstInt,
    MAX_OBS: ct.ConstInt,
) -> None:
    """Evaluate 3D SDF distance, collision barrier, and surface normals for a tile of query points."""
    bid_k = ct.bid(0)

    pts_tile = ct.load(points_arr, index=(bid_k, 0), shape=(TILE_K, 3))
    obs_tile = ct.load(obs_table_arr, index=(0, 0), shape=(MAX_OBS, 16))
    params_tile = ct.load(params_arr, index=(0,), shape=(8,))

    t_val = ct.extract(params_tile, index=(0,), shape=(1,))
    r_robot = ct.extract(params_tile, index=(1,), shape=(1,))
    m_safe = ct.extract(params_tile, index=(2,), shape=(1,))
    w_hard = ct.extract(params_tile, index=(3,), shape=(1,))
    w_soft = ct.extract(params_tile, index=(4,), shape=(1,))
    alpha = ct.extract(params_tile, index=(5,), shape=(1,))
    eps_val = ct.extract(params_tile, index=(6,), shape=(1,))

    d_center = eval_sdf_tile_block(pts_tile, obs_tile, t_val, TILE_K, MAX_OBS)
    barrier = eval_barrier_cost_tile_block(d_center, r_robot, m_safe, w_hard, w_soft, alpha, TILE_K)

    eps_b = ct.broadcast_to(ct.reshape(eps_val, (1, 1)), (TILE_K, 1))
    zero_col = ct.zeros((TILE_K, 1), dtype=ct.float32)

    dx_vec = ct.cat([eps_b, zero_col, zero_col], axis=-1)
    dy_vec = ct.cat([zero_col, eps_b, zero_col], axis=-1)
    dz_vec = ct.cat([zero_col, zero_col, eps_b], axis=-1)

    gx = eval_sdf_tile_block(pts_tile + dx_vec, obs_tile, t_val, TILE_K, MAX_OBS) - (
        eval_sdf_tile_block(pts_tile - dx_vec, obs_tile, t_val, TILE_K, MAX_OBS)
    )
    gy = eval_sdf_tile_block(pts_tile + dy_vec, obs_tile, t_val, TILE_K, MAX_OBS) - (
        eval_sdf_tile_block(pts_tile - dy_vec, obs_tile, t_val, TILE_K, MAX_OBS)
    )
    gz = eval_sdf_tile_block(pts_tile + dz_vec, obs_tile, t_val, TILE_K, MAX_OBS) - (
        eval_sdf_tile_block(pts_tile - dz_vec, obs_tile, t_val, TILE_K, MAX_OBS)
    )

    two_eps = ct.full((TILE_K, 1), 2.0, dtype=ct.float32) * eps_b
    gx = gx / two_eps
    gy = gy / two_eps
    gz = gz / two_eps

    norm_sq = gx * gx + gy * gy + gz * gz
    tiny = ct.full((TILE_K, 1), 1e-14, dtype=ct.float32)
    inv_norm = ct.rsqrt(ct.maximum(norm_sq, tiny))

    nx = ct.where(norm_sq > tiny, gx * inv_norm, zero_col)
    ny = ct.where(norm_sq > tiny, gy * inv_norm, zero_col)
    nz = ct.where(norm_sq > tiny, gz * inv_norm, ct.ones((TILE_K, 1), dtype=ct.float32))
    normals_tile = ct.cat([nx, ny, nz], axis=-1)

    ct.store(out_sdf_arr, index=(bid_k,), tile=ct.reshape(d_center, (TILE_K,)))
    ct.store(out_barrier_arr, index=(bid_k,), tile=ct.reshape(barrier, (TILE_K,)))
    ct.store(out_normals_arr, index=(bid_k, 0), tile=normals_tile)


def launch_tiled_sdf_cutile(
    points: Float32Array,
    obstacles_table: Float32Array,
    t: float,
    sdf_cfg: SDFConfig,
    tile_k: int = 128,
    eps: float = 1e-4,
    prefer_native: bool = False,
) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Host launcher for tiled_sdf_collision_kernel."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n_pts = pts.shape[0]
    max_obs = obstacles_table.shape[0]

    out_sdf = np.zeros(n_pts, dtype=np.float32)
    out_barrier = np.zeros(n_pts, dtype=np.float32)
    out_normals = np.zeros((n_pts, 3), dtype=np.float32)

    params_arr = np.array(
        [
            t,
            sdf_cfg.robot_radius,
            sdf_cfg.safety_margin,
            sdf_cfg.hard_barrier_weight,
            sdf_cfg.soft_barrier_weight,
            sdf_cfg.soft_decay_alpha,
            eps,
            0.0,
        ],
        dtype=np.float32,
    )

    grid = (ct.cdiv(n_pts, tile_k), 1, 1)
    ct.launch(
        None,
        grid,
        tiled_sdf_collision_kernel,
        (
            pts,
            np.asarray(obstacles_table, dtype=np.float32),
            params_arr,
            out_sdf,
            out_barrier,
            out_normals,
            tile_k,
            max_obs,
        ),
        prefer_native=prefer_native,
    )
    return out_sdf, out_barrier, out_normals

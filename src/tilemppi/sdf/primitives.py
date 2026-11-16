"""Exact 3D Signed Distance Function primitives, dynamic kinematics, and barrier costs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from tilemppi.config import SDFConfig
from tilemppi.types import Float32Array, ObstacleKind, is_power_of_two

INACTIVE_OBSTACLE_KIND = -1.0
FAR_SDF_DISTANCE = 1.0e4


def canonicalize_unit_quat(quat_wxyz: Sequence[float] | Float32Array) -> Float32Array:
    """Normalize [qw, qx, qy, qz] to unit length with qw >= 0."""
    q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q = q / norm
    if q[0] < 0.0:
        q = -q
    return q.astype(np.float32)


@dataclass(slots=True)
class ObstaclePrimitive:
    """High-level representation of a static or dynamic 3D SDF obstacle."""

    kind: ObstacleKind
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    params: tuple[float, float, float] = (0.5, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    bounce_in_arena: bool = False
    label: str = field(default="obstacle")

    @classmethod
    def sphere(
        cls,
        position: tuple[float, float, float],
        radius: float,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0),
        bounce_in_arena: bool = False,
        label: str = "sphere",
    ) -> ObstaclePrimitive:
        """Create a 3D sphere obstacle."""
        return cls(
            kind=ObstacleKind.SPHERE,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            params=(radius, 0.0, 0.0),
            bounce_in_arena=bounce_in_arena,
            label=label,
        )

    @classmethod
    def capsule(
        cls,
        position: tuple[float, float, float],
        radius: float,
        half_height: float,
        quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        label: str = "capsule",
    ) -> ObstaclePrimitive:
        """Create a 3D capsule obstacle aligned with its local Z axis."""
        return cls(
            kind=ObstacleKind.CAPSULE,
            position=position,
            velocity=velocity,
            params=(radius, half_height, 0.0),
            quat_wxyz=quat_wxyz,
            label=label,
        )

    @classmethod
    def oriented_box(
        cls,
        position: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        label: str = "box",
    ) -> ObstaclePrimitive:
        """Create a quaternion-oriented 3D bounding box (OBB) obstacle."""
        return cls(
            kind=ObstacleKind.ORIENTED_BOX,
            position=position,
            velocity=velocity,
            params=half_extents,
            quat_wxyz=quat_wxyz,
            label=label,
        )

    @classmethod
    def torus_gate(
        cls,
        position: tuple[float, float, float],
        major_radius: float,
        tube_radius: float,
        quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0),
        label: str = "torus_gate",
    ) -> ObstaclePrimitive:
        """Create a fly-through torus ring gate (ring in local YZ plane, open along local X)."""
        return cls(
            kind=ObstacleKind.TORUS_GATE,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            params=(major_radius, tube_radius, 0.0),
            quat_wxyz=quat_wxyz,
            label=label,
        )

    @classmethod
    def ground_plane(
        cls,
        z_height: float = 0.0,
        label: str = "ground",
    ) -> ObstaclePrimitive:
        """Create an infinite horizontal ground plane z = z_height."""
        return cls(
            kind=ObstacleKind.GROUND_PLANE,
            position=(0.0, 0.0, z_height),
            params=(0.0, 0.0, 0.0),
            label=label,
        )

    def to_row16(self) -> Float32Array:
        """Pack primitive into a 16-float tile-aligned vector."""
        row = np.zeros(16, dtype=np.float32)
        q = canonicalize_unit_quat(self.quat_wxyz)
        row[0] = float(int(self.kind))
        row[1:4] = np.asarray(self.position, dtype=np.float32)
        row[4:7] = np.asarray(self.velocity, dtype=np.float32)
        row[7:10] = np.asarray(self.acceleration, dtype=np.float32)
        row[10:13] = np.asarray(self.params, dtype=np.float32)
        row[13:16] = q[1:4]  # [qx, qy, qz], with qw = sqrt(max(0, 1 - |q_xyz|^2))
        return row


def pack_obstacles_table(
    obstacles: Sequence[ObstaclePrimitive],
    max_obstacles: int = 32,
) -> Float32Array:
    """Pack obstacle list into a power-of-two (MAX_OBS, 16) float32 table."""
    if not is_power_of_two(max_obstacles):
        msg = f"max_obstacles must be a power of two, got {max_obstacles}"
        raise ValueError(msg)

    table = np.zeros((max_obstacles, 16), dtype=np.float32)
    table[:, 0] = INACTIVE_OBSTACLE_KIND

    count = min(len(obstacles), max_obstacles)
    for idx in range(count):
        table[idx] = obstacles[idx].to_row16()
    return table


def predict_obstacle_positions(table: Float32Array, t: float) -> Float32Array:
    """Predict obstacle centers p(t) = p0 + v*t + 0.5*a*t^2 for all rows in table."""
    p0 = table[:, 1:4]
    vel = table[:, 4:7]
    acc = table[:, 7:10]
    t_f = np.float32(t)
    return np.asarray(p0 + vel * t_f + np.float32(0.5) * acc * (t_f * t_f), dtype=np.float32)


def step_dynamic_obstacles_in_place(
    table: Float32Array,
    dt: float,
    arena_bounds: tuple[float, float, float] = (8.0, 8.0, 5.0),
) -> None:
    """Advance dynamic obstacles by dt in world time with smooth arena reflection."""
    active_mask = table[:, 0] >= 0.0
    if not np.any(active_mask):
        return

    dt_f = np.float32(dt)
    pos = table[active_mask, 1:4]
    vel = table[active_mask, 4:7]
    acc = table[active_mask, 7:10]

    new_pos = pos + vel * dt_f + np.float32(0.5) * acc * (dt_f * dt_f)
    new_vel = vel + acc * dt_f

    bx, by, bz = arena_bounds
    limits = np.array([bx, by, bz], dtype=np.float32)
    lower = np.array([-bx, -by, 0.25], dtype=np.float32)

    for axis in range(3):
        over = new_pos[:, axis] > limits[axis]
        under = new_pos[:, axis] < lower[axis]
        if np.any(over):
            new_pos[over, axis] = limits[axis]
            new_vel[over, axis] = -np.abs(new_vel[over, axis])
        if np.any(under):
            new_pos[under, axis] = lower[axis]
            new_vel[under, axis] = np.abs(new_vel[under, axis])

    table[active_mask, 1:4] = new_pos
    table[active_mask, 4:7] = new_vel


def _rotate_world_to_local(diff: Float32Array, q_xyz: Float32Array) -> Float32Array:
    """Rotate N x 3 displacement vectors by the inverse (conjugate) of unit quaternion q."""
    qx = np.float32(q_xyz[0])
    qy = np.float32(q_xyz[1])
    qz = np.float32(q_xyz[2])
    qw_sq = np.maximum(np.float32(0.0), np.float32(1.0) - (qx * qx + qy * qy + qz * qz))
    qw = np.sqrt(qw_sq)

    # Conjugate quaternion q* = (qw, -qx, -qy, -qz)
    # v' = v + 2 * qw * (q_vec* x v) + 2 * (q_vec* x (q_vec* x v))
    ux, uy, uz = -qx, -qy, -qz
    vx = diff[:, 0]
    vy = diff[:, 1]
    vz = diff[:, 2]

    tx = np.float32(2.0) * (uy * vz - uz * vy)
    ty = np.float32(2.0) * (uz * vx - ux * vz)
    tz = np.float32(2.0) * (ux * vy - uy * vx)

    rx = vx + qw * tx + (uy * tz - uz * ty)
    ry = vy + qw * ty + (uz * tx - ux * tz)
    rz = vz + qw * tz + (ux * ty - uy * tx)
    return np.asarray(np.stack((rx, ry, rz), axis=-1), dtype=np.float32)


def evaluate_single_primitive_sdf(
    points: Float32Array,
    row: Float32Array,
    t: float = 0.0,
) -> Float32Array:
    """Evaluate exact 3D Signed Distance Function of a single obstacle row at points (N, 3)."""
    pts = np.asarray(points, dtype=np.float32)
    orig_shape = pts.shape[:-1]
    pts_flat = pts.reshape(-1, 3)
    n_pts = pts_flat.shape[0]

    kind_val = round(float(row[0]))
    if kind_val < 0:
        return np.full(orig_shape, FAR_SDF_DISTANCE, dtype=np.float32)

    t_f = np.float32(t)
    center = row[1:4] + row[4:7] * t_f + np.float32(0.5) * row[7:10] * (t_f * t_f)
    params = row[10:13]
    q_xyz = row[13:16]

    if kind_val == int(ObstacleKind.GROUND_PLANE):
        dist = pts_flat[:, 2] - (center[2] + params[0])
        return np.asarray(dist.reshape(orig_shape), dtype=np.float32)

    diff = pts_flat - center.reshape(1, 3)
    local_p = _rotate_world_to_local(diff, q_xyz)

    if kind_val == int(ObstacleKind.SPHERE):
        radius = np.float32(params[0])
        dist = np.linalg.norm(local_p, axis=-1) - radius
    elif kind_val == int(ObstacleKind.CAPSULE):
        radius = np.float32(params[0])
        half_h = np.float32(params[1])
        clamped_z = np.clip(local_p[:, 2], -half_h, half_h)
        dz = local_p[:, 2] - clamped_z
        dist = np.sqrt(local_p[:, 0] ** 2 + local_p[:, 1] ** 2 + dz * dz) - radius
    elif kind_val == int(ObstacleKind.ORIENTED_BOX):
        bx, by, bz = np.float32(params[0]), np.float32(params[1]), np.float32(params[2])
        qx_b = np.abs(local_p[:, 0]) - bx
        qy_b = np.abs(local_p[:, 1]) - by
        qz_b = np.abs(local_p[:, 2]) - bz
        out_x = np.maximum(qx_b, np.float32(0.0))
        out_y = np.maximum(qy_b, np.float32(0.0))
        out_z = np.maximum(qz_b, np.float32(0.0))
        outside_dist = np.sqrt(out_x * out_x + out_y * out_y + out_z * out_z)
        inside_dist = np.minimum(np.maximum(qx_b, np.maximum(qy_b, qz_b)), np.float32(0.0))
        dist = outside_dist + inside_dist
    elif kind_val == int(ObstacleKind.TORUS_GATE):
        major_r = np.float32(params[0])
        tube_r = np.float32(params[1])
        rho_yz = np.sqrt(local_p[:, 1] ** 2 + local_p[:, 2] ** 2)
        dr = rho_yz - major_r
        dist = np.sqrt(local_p[:, 0] ** 2 + dr * dr) - tube_r
    else:
        dist = np.full(n_pts, FAR_SDF_DISTANCE, dtype=np.float32)

    return np.asarray(dist.reshape(orig_shape), dtype=np.float32)


def smooth_min_sdf(
    d1: Float32Array,
    d2: Float32Array,
    k: float = 12.0,
) -> Float32Array:
    """Exponential/polynomial smooth minimum of two signed distance fields."""
    k_f = np.float32(max(k, 1e-3))
    h = np.maximum(k_f - np.abs(d1 - d2), np.float32(0.0)) / k_f
    return np.asarray(np.minimum(d1, d2) - h * h * k_f * np.float32(0.25), dtype=np.float32)


def evaluate_sdf_batch(
    points: Float32Array,
    obstacles_table: Float32Array,
    t: float = 0.0,
    use_smooth_min: bool = False,
    smooth_k: float = 12.0,
) -> Float32Array:
    """Evaluate minimum signed distance from points (..., 3) across all active obstacles."""
    pts = np.asarray(points, dtype=np.float32)
    orig_shape = pts.shape[:-1]
    pts_flat = pts.reshape(-1, 3)

    min_d = np.full(pts_flat.shape[0], FAR_SDF_DISTANCE, dtype=np.float32)
    smooth_param = np.float32(1.0 / max(smooth_k, 0.5))

    for row in obstacles_table:
        if row[0] < 0.0:
            continue
        d_obs = evaluate_single_primitive_sdf(pts_flat, row, t=t)
        if use_smooth_min:
            min_d = smooth_min_sdf(min_d, d_obs, k=float(smooth_param))
        else:
            min_d = np.minimum(min_d, d_obs)

    return min_d.reshape(orig_shape).astype(np.float32)


def evaluate_sdf_and_normals(
    points: Float32Array,
    obstacles_table: Float32Array,
    t: float = 0.0,
    eps: float = 1e-4,
) -> tuple[Float32Array, Float32Array]:
    """Compute signed distance and unit surface normals via central differences."""
    pts = np.asarray(points, dtype=np.float32)
    orig_shape = pts.shape[:-1]
    pts_flat = pts.reshape(-1, 3)

    sdf_val = evaluate_sdf_batch(pts_flat, obstacles_table, t=t)
    delta = np.float32(eps)

    dx_pos = pts_flat.copy()
    dx_neg = pts_flat.copy()
    dx_pos[:, 0] += delta
    dx_neg[:, 0] -= delta
    gx = evaluate_sdf_batch(dx_pos, obstacles_table, t=t) - evaluate_sdf_batch(
        dx_neg, obstacles_table, t=t
    )

    dy_pos = pts_flat.copy()
    dy_neg = pts_flat.copy()
    dy_pos[:, 1] += delta
    dy_neg[:, 1] -= delta
    gy = evaluate_sdf_batch(dy_pos, obstacles_table, t=t) - evaluate_sdf_batch(
        dy_neg, obstacles_table, t=t
    )

    dz_pos = pts_flat.copy()
    dz_neg = pts_flat.copy()
    dz_pos[:, 2] += delta
    dz_neg[:, 2] -= delta
    gz = evaluate_sdf_batch(dz_pos, obstacles_table, t=t) - evaluate_sdf_batch(
        dz_neg, obstacles_table, t=t
    )

    grad = np.stack((gx, gy, gz), axis=-1) / (np.float32(2.0) * delta)
    norms = np.linalg.norm(grad, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 1e-7, norms, np.float32(1.0))
    unit_normals = np.where(
        norms > 1e-7,
        grad / safe_norms,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )

    return (
        sdf_val.reshape(orig_shape).astype(np.float32),
        unit_normals.reshape((*orig_shape, 3)).astype(np.float32),
    )


def compute_barrier_cost(
    sdf_values: Float32Array,
    sdf_cfg: SDFConfig,
    cvar_uncertainty_scale: float = 0.0,
) -> Float32Array:
    """Compute smooth differentiable collision barrier cost from SDF distances.

    phi_coll(d) = w_hard * max(0, r_robot + m_safe + sigma_cvar - d)^2
                + w_soft * exp(-alpha * (d - r_robot))
    """
    d = np.asarray(sdf_values, dtype=np.float32)
    r_robot = np.float32(sdf_cfg.robot_radius)
    m_safe = np.float32(sdf_cfg.safety_margin + cvar_uncertainty_scale)
    w_hard = np.float32(sdf_cfg.hard_barrier_weight)
    w_soft = np.float32(sdf_cfg.soft_barrier_weight)
    alpha = np.float32(sdf_cfg.soft_decay_alpha)

    penetration = np.maximum(np.float32(0.0), (r_robot + m_safe) - d)
    hard_cost = w_hard * (penetration * penetration)

    exponent = np.clip(-alpha * (d - r_robot), np.float32(-30.0), np.float32(15.0))
    soft_cost = w_soft * np.exp(exponent)
    return np.asarray(hard_cost + soft_cost, dtype=np.float32)

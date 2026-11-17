"""High-throughput PyTorch CUDA (Ada sm_89 Tensor Core FP32/FP16/FP8) and CPU MPPI backend."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from tilemppi.config import MPPIConfig, SDFConfig
from tilemppi.dynamics import create_robot_model
from tilemppi.dynamics.quadruped import NOMINAL_HIP_FLEX, NOMINAL_KNEE_FLEX, NOMINAL_STANDING_Z
from tilemppi.dynamics.swarm import SWARM_FORMATION_OFFSETS
from tilemppi.kernels.reference_cpu import rollout_single_trajectory_xyz
from tilemppi.types import Float32Array, PrecisionMode, RobotKind

torch: Any
try:
    import torch as _torch_mod

    torch = _torch_mod
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    _TORCH_AVAILABLE = False


def is_torch_available() -> bool:
    """Return True if PyTorch is installed."""
    return _TORCH_AVAILABLE


def _quantize_torch_tile(tensor: Any, precision: PrecisionMode) -> Any:
    """Apply FP16 or native Ada sm_89 FP8 (float8_e4m3fn) quantization to a PyTorch tensor."""
    if precision == PrecisionMode.FP32:
        return tensor.to(torch.float32)
    if precision == PrecisionMode.FP16:
        return tensor.to(torch.float16).to(torch.float32)
    # FP8 E4M3FN: use native torch.float8_e4m3fn when supported or exact E4M3 quantization
    clipped = torch.clamp(tensor.to(torch.float32), -448.0, 448.0)
    abs_val = torch.abs(clipped)
    sign = torch.sign(clipped)
    nonzero = abs_val > 1e-6
    safe_abs = torch.where(nonzero, abs_val, torch.ones_like(abs_val))
    exp2 = torch.floor(torch.log2(safe_abs))
    scale = torch.pow(2.0, exp2 - 3.0)
    safe_scale = torch.clamp(scale, min=1e-5)
    q = sign * torch.round(abs_val / safe_scale) * safe_scale
    return torch.where(nonzero, q, torch.zeros_like(q))


def generate_philox_ou_noise_torch(
    num_rollouts: int,
    horizon: int,
    control_dim: int,
    sigma_vec: Any,
    ou_rho: float,
    seed: int | Any,
    precision: PrecisionMode,
    device: Any,
) -> Any:
    """Synthesize (K, H, Du) Philox + Box-Muller + OU colored noise directly on GPU/CPU device."""
    mask32 = 0xFFFFFFFF
    mask24 = 0x00FFFFFF

    k_idx = torch.arange(num_rollouts, dtype=torch.int64, device=device).view(-1, 1)
    d_idx = torch.arange(control_dim, dtype=torch.int64, device=device).view(1, -1)
    if isinstance(seed, int):
        seed_t = torch.tensor(seed & mask32, dtype=torch.int64, device=device)
    else:
        seed_t = (seed & mask32).to(torch.int64)

    c0 = (k_idx ^ ((seed_t + 0x9E3779B9) & mask32)) & mask32
    d_term = ((d_idx * 1013904223) & mask32) ^ ((seed_t * 2246822519) & mask32)

    sigma_t = sigma_vec.view(1, control_dim).to(device=device, dtype=torch.float32)
    rho_f = float(np.float32(ou_rho))
    ou_scale_f = float(
        np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - np.float32(rho_f * rho_f)))
    )

    inv_24 = float(np.float32(1.0 / 16777217.0))
    two_pi_inv_24 = float(np.float32(6.283185307179586) * np.float32(1.0 / 16777217.0))

    out = torch.empty((num_rollouts, horizon, control_dim), dtype=torch.float32, device=device)
    prev = torch.zeros((num_rollouts, control_dim), dtype=torch.float32, device=device)

    for t in range(horizon):
        t_term = (t * 1664525) & mask32
        c1 = (t_term ^ d_term) & mask32

        h1 = ((c0 ^ (c1 >> 15)) * 0x85EBCA6B) & mask32
        h1 = ((h1 ^ (h1 >> 13)) * 0xC2B2AE35) & mask32
        u1_bits = (h1 ^ (h1 >> 16)) & mask32

        h2 = ((c1 ^ (c0 >> 13) ^ 0x27D4EB2D) * 0x165667B1) & mask32
        h2 = ((h2 ^ (h2 >> 15)) * 0x85EBCA6B) & mask32
        u2_bits = (h2 ^ (h2 >> 16)) & mask32

        m1 = (u1_bits & mask24).to(torch.float32)
        m2 = (u2_bits & mask24).to(torch.float32)

        u1 = (m1 + 1.0) * inv_24
        u2 = (m2 + 1.0) * two_pi_inv_24

        z_t = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(u2)
        scaled = z_t * sigma_t
        curr = scaled if t == 0 else rho_f * prev + ou_scale_f * scaled
        curr_q = _quantize_torch_tile(curr, precision)
        out[:, t, :] = curr_q
        prev = curr_q

    return out


def evaluate_sdf_batch_torch(
    points: Any,
    obstacles_table: Any,
    t: float,
    assume_all_active: bool = False,
    static_table: bool = False,
) -> Any:
    """Vectorized 3D SDF evaluation on GPU/CPU tensor (K, 3) against active obstacle rows."""
    if static_table:
        active_obs = obstacles_table
    elif assume_all_active:
        active_obs = obstacles_table
        if active_obs.shape[0] == 0:
            return torch.full((points.shape[0],), 1.0e4, dtype=torch.float32, device=points.device)
    else:
        active_mask = obstacles_table[:, 0] >= 0.0
        if not bool(torch.any(active_mask)):
            return torch.full((points.shape[0],), 1.0e4, dtype=torch.float32, device=points.device)
        active_obs = obstacles_table[active_mask]  # (M_act, 16)
        if active_obs.shape[0] == 0:
            return torch.full((points.shape[0],), 1.0e4, dtype=torch.float32, device=points.device)

    kinds = torch.round(active_obs[:, 0]).to(torch.int64)  # (M_act,)

    t_f = float(t)
    t2_half = 0.5 * t_f * t_f
    centers = active_obs[:, 1:4] + active_obs[:, 4:7] * t_f + active_obs[:, 7:10] * t2_half
    params = active_obs[:, 10:13]
    q_xyz = active_obs[:, 13:16]

    qx = q_xyz[:, 0].unsqueeze(0)
    qy = q_xyz[:, 1].unsqueeze(0)
    qz = q_xyz[:, 2].unsqueeze(0)
    qw = torch.sqrt(torch.clamp(1.0 - (qx * qx + qy * qy + qz * qz), min=0.0))

    # Broadcast diff: (K, M_act, 3)
    diff = points.unsqueeze(1) - centers.unsqueeze(0)
    vx = diff[:, :, 0]
    vy = diff[:, :, 1]
    vz = diff[:, :, 2]

    ux, uy, uz = -qx, -qy, -qz
    tx = 2.0 * (uy * vz - uz * vy)
    ty = 2.0 * (uz * vx - ux * vz)
    tz = 2.0 * (ux * vy - uy * vx)

    lx = vx + qw * tx + (uy * tz - uz * ty)
    ly = vy + qw * ty + (uz * tx - ux * tz)
    lz = vz + qw * tz + (ux * ty - uy * tx)

    p0 = params[:, 0].unsqueeze(0)
    p1 = params[:, 1].unsqueeze(0)
    p2 = params[:, 2].unsqueeze(0)

    # 0: Sphere
    d_sphere = torch.sqrt(lx * lx + ly * ly + lz * lz) - p0

    # 1: Capsule
    cz_clamp = torch.clamp(lz, -p1, p1)
    dz_cap = lz - cz_clamp
    d_capsule = torch.sqrt(lx * lx + ly * ly + dz_cap * dz_cap) - p0

    # 2: OrientedBox
    bx = torch.abs(lx) - p0
    by = torch.abs(ly) - p1
    bz = torch.abs(lz) - p2
    ox_b = torch.clamp(bx, min=0.0)
    oy_b = torch.clamp(by, min=0.0)
    oz_b = torch.clamp(bz, min=0.0)
    d_box = torch.sqrt(ox_b * ox_b + oy_b * oy_b + oz_b * oz_b) + torch.clamp(
        torch.maximum(bx, torch.maximum(by, bz)), max=0.0
    )

    # 3: TorusGate
    rho_yz = torch.sqrt(ly * ly + lz * lz) - p0
    d_torus = torch.sqrt(lx * lx + rho_yz * rho_yz) - p1

    # 4: GroundPlane
    d_ground = points[:, 2:3] - (centers[:, 2].unsqueeze(0) + p0)

    k_exp = kinds.unsqueeze(0)
    d_all = torch.where(
        k_exp < 0,
        torch.full_like(d_sphere, 1.0e4),
        torch.where(
            k_exp == 0,
            d_sphere,
            torch.where(
                k_exp == 1,
                d_capsule,
                torch.where(
                    k_exp == 2,
                    d_box,
                    torch.where(k_exp == 3, d_torus, d_ground),
                ),
            ),
        ),
    )
    min_d, _ = torch.min(d_all, dim=1)
    return min_d


def compute_barrier_cost_torch(
    sdf_vals: Any,
    sdf_cfg: SDFConfig,
    cvar_scale: float = 0.0,
) -> Any:
    """Compute differentiable collision barrier cost on PyTorch tensor."""
    r_robot = float(sdf_cfg.robot_radius)
    m_safe = float(sdf_cfg.safety_margin + cvar_scale)
    w_hard = float(sdf_cfg.hard_barrier_weight)
    w_soft = float(sdf_cfg.soft_barrier_weight)
    alpha = float(sdf_cfg.soft_decay_alpha)

    pen = torch.clamp((r_robot + m_safe) - sdf_vals, min=0.0)
    hard_c = w_hard * (pen * pen)
    exp_arg = torch.clamp(-alpha * (sdf_vals - r_robot), min=-30.0, max=15.0)
    soft_c = w_soft * torch.exp(exp_arg)
    return hard_c + soft_c


def _quadrotor_deriv_torch(x: Any, u: Any) -> Any:
    dx = torch.zeros_like(x)
    qw, qx, qy, qz = x[:, 3], x[:, 4], x[:, 5], x[:, 6]
    vx, vy, vz = x[:, 7], x[:, 8], x[:, 9]
    wx, wy, wz = x[:, 10], x[:, 11], x[:, 12]
    f0, f1, f2, f3 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

    arm_eff = float(0.18 / np.sqrt(2.0))
    yaw_c = 0.022
    ixx, iyy, izz = 0.0085, 0.0085, 0.015
    d_ang = 0.08

    total_thrust = f0 + f1 + f2 + f3
    tau_x = arm_eff * (f0 - f1 - f2 + f3) - 0.35 * qx
    tau_y = arm_eff * (f0 + f1 - f2 - f3) - 0.35 * qy
    tau_z = yaw_c * (f0 - f1 + f2 - f3) - 0.15 * qz

    bz_x = 2.0 * (qx * qz + qw * qy)
    bz_y = 2.0 * (qy * qz - qw * qx)
    bz_z = qw * qw - qx * qx - qy * qy + qz * qz

    dx[:, 0] = vx
    dx[:, 1] = vy
    dx[:, 2] = vz
    dx[:, 3] = 0.5 * (-qx * wx - qy * wy - qz * wz)
    dx[:, 4] = 0.5 * (qw * wx + qy * wz - qz * wy)
    dx[:, 5] = 0.5 * (qw * wy - qx * wz + qz * wx)
    dx[:, 6] = 0.5 * (qw * wz + qx * wy - qy * wx)
    dx[:, 7] = total_thrust * bz_x
    dx[:, 8] = total_thrust * bz_y
    dx[:, 9] = total_thrust * bz_z - 9.81
    dx[:, 10] = (tau_x - (izz - iyy) * wy * wz - d_ang * wx) / ixx
    dx[:, 11] = (tau_y - (ixx - izz) * wz * wx - d_ang * wy) / iyy
    dx[:, 12] = (tau_z - (iyy - ixx) * wx * wy - d_ang * wz) / izz
    dx[:, 13] = 0.5 * (total_thrust - 9.81)
    return dx


def _quadrotor_project_torch(x: Any) -> Any:
    out = x.clone()
    q = out[:, 3:7]
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    out[:, 3:7] = q / torch.clamp(norm, min=1e-8)
    return out


def _quadruped_deriv_torch(x: Any, u: Any, nom_joints: Any) -> Any:
    dx = torch.zeros_like(x)
    pz = x[:, 2]
    qw, qx, qy, qz = x[:, 3], x[:, 4], x[:, 5], x[:, 6]
    vx, vy, vz = x[:, 7], x[:, 8], x[:, 9]
    wx, wy, wz = x[:, 10], x[:, 11], x[:, 12]
    phi = x[:, 13]

    vx_cmd, vy_cmd, wz_cmd, stance_boost = u[:, 12], u[:, 13], u[:, 14], u[:, 15]

    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    j_stride_x = 0.25 * (u[:, 1] + u[:, 4] + u[:, 7] + u[:, 10])
    j_stride_y = 0.25 * (u[:, 0] + u[:, 3] + u[:, 6] + u[:, 9])
    v_bx = vx_cmd + 0.8 * j_stride_x
    v_by = vy_cmd + 0.5 * j_stride_y

    v_dx = cy * v_bx - sy * v_by
    v_dy = sy * v_bx + cy * v_by

    c_gate = torch.clamp((0.58 - pz) / 0.16, 0.0, 1.0)
    k_trac = 95.0 * c_gate
    fx = k_trac * (v_dx - vx)
    fy = k_trac * (v_dy - vy)
    fz = torch.clamp(
        c_gate
        * (12.0 * 9.81 + 380.0 * (NOMINAL_STANDING_Z - pz) - 48.0 * vz + 18.0 * stance_boost),
        min=0.0,
    )

    inv_m = 1.0 / 12.0
    dx[:, 0] = vx
    dx[:, 1] = vy
    dx[:, 2] = vz
    dx[:, 3] = 0.5 * (-qx * wx - qy * wy - qz * wz)
    dx[:, 4] = 0.5 * (qw * wx + qy * wz - qz * wy)
    dx[:, 5] = 0.5 * (qw * wy - qx * wz + qz * wx)
    dx[:, 6] = 0.5 * (qw * wz + qx * wy - qy * wx)
    dx[:, 7] = fx * inv_m
    dx[:, 8] = fy * inv_m
    dx[:, 9] = fz * inv_m - 9.81
    dx[:, 10] = (-65.0 * qx - 14.0 * wx) / 0.14
    dx[:, 11] = (-65.0 * qy - 14.0 * wy) / 0.42
    dx[:, 12] = (8.5 * c_gate * (wz_cmd - wz) - 1.5 * wz) / 0.48

    cmd_speed = torch.sqrt(v_bx * v_bx + v_by * v_by + wz_cmd * wz_cmd)
    g_gate = torch.clamp(cmd_speed / 0.5, 0.0, 1.0)
    dx[:, 13] = 8.0 * g_gate
    s_trot = torch.sin(phi) * g_gate

    for leg in range(4):
        p_sign = 1.0 if leg in (0, 3) else -1.0
        b_idx = 14 + 3 * leg
        u_idx = 3 * leg
        q_abd_t = nom_joints[u_idx] + 0.35 * u[:, u_idx]
        q_hip_t = nom_joints[u_idx + 1] + 0.28 * p_sign * s_trot + 0.45 * u[:, u_idx + 1]
        q_knee_t = (
            nom_joints[u_idx + 2]
            - 0.35 * torch.clamp(p_sign * s_trot, min=0.0)
            + 0.45 * u[:, u_idx + 2]
        )
        dx[:, b_idx] = 12.0 * (q_abd_t - x[:, b_idx])
        dx[:, b_idx + 1] = 12.0 * (q_hip_t - x[:, b_idx + 1])
        dx[:, b_idx + 2] = 12.0 * (q_knee_t - x[:, b_idx + 2])
        c_targ = c_gate * torch.where(
            p_sign * s_trot <= 0.15, torch.ones_like(s_trot), torch.zeros_like(s_trot)
        )
        dx[:, 26 + leg] = 10.0 * (c_targ - x[:, 26 + leg])

    return dx


def _quadruped_project_torch(x: Any) -> Any:
    out = x.clone()
    q = out[:, 3:7]
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    out[:, 3:7] = q / torch.clamp(norm, min=1e-8)
    out[:, 2] = torch.clamp(out[:, 2], min=0.12)
    out[:, 14:26] = torch.clamp(out[:, 14:26], -2.2, 2.2)
    out[:, 26:30] = torch.clamp(out[:, 26:30], 0.0, 1.0)
    return out


def _swarm_deriv_torch(x: Any, u: Any) -> Any:
    dx = torch.zeros_like(x)
    pos = [x[:, 8 * i : 8 * i + 3] for i in range(4)]
    rep = [torch.zeros_like(pos[0]) for _ in range(4)]

    for i in range(4):
        for j in range(i + 1, 4):
            diff = pos[i] - pos[j]
            dist = torch.sqrt(torch.sum(diff * diff, dim=-1, keepdim=True) + 1e-6)
            overlap = torch.clamp(0.52 - dist, min=0.0)
            f_vec = (12.0 * overlap / dist) * diff
            rep[i] = rep[i] + f_vec
            rep[j] = rep[j] - f_vec

    inv_tau = 1.0 / 0.12
    for i in range(4):
        bs = 8 * i
        bu = 4 * i
        vx, vy, vz = x[:, bs + 3], x[:, bs + 4], x[:, bs + 5]
        roll, pitch = x[:, bs + 6], x[:, bs + 7]
        thrust, r_cmd, p_cmd, c_trim = u[:, bu], u[:, bu + 1], u[:, bu + 2], u[:, bu + 3]

        dx[:, bs] = vx
        dx[:, bs + 1] = vy
        dx[:, bs + 2] = vz
        dx[:, bs + 3] = thrust * torch.sin(pitch) - 0.25 * vx + rep[i][:, 0]
        dx[:, bs + 4] = -thrust * torch.sin(roll) * torch.cos(pitch) - 0.25 * vy + rep[i][:, 1]
        dx[:, bs + 5] = (
            thrust * torch.cos(roll) * torch.cos(pitch)
            - 9.81
            + 0.4 * c_trim
            - 0.25 * vz
            + rep[i][:, 2]
        )
        dx[:, bs + 6] = inv_tau * (r_cmd - roll)
        dx[:, bs + 7] = inv_tau * (p_cmd - pitch)
    return dx


def _swarm_project_torch(x: Any) -> Any:
    out = x.clone()
    for i in range(4):
        out[:, 8 * i + 2] = torch.clamp(out[:, 8 * i + 2], min=0.20)
        out[:, 8 * i + 6] = torch.clamp(out[:, 8 * i + 6], -0.85, 0.85)
        out[:, 8 * i + 7] = torch.clamp(out[:, 8 * i + 7], -0.85, 0.85)
    return out


class TorchMPPIBackend:
    """Fused PyTorch GPU (CUDA sm_89 Tensor Core) and CPU MPPI backend."""

    def __init__(
        self,
        robot: RobotKind,
        mppi_cfg: MPPIConfig,
        sdf_cfg: SDFConfig,
        precision: PrecisionMode = PrecisionMode.FP32,
        use_cuda: bool = True,
    ) -> None:
        if not _TORCH_AVAILABLE:
            msg = "PyTorch is not installed"
            raise RuntimeError(msg)

        self.robot_kind = robot
        self.model = create_robot_model(robot)
        self.mppi_cfg = mppi_cfg
        self.sdf_cfg = sdf_cfg
        self.precision = precision
        self.device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")

        self._u_min = torch.as_tensor(
            self.model.control_min, dtype=torch.float32, device=self.device
        )
        self._u_max = torch.as_tensor(
            self.model.control_max, dtype=torch.float32, device=self.device
        )
        self._u_nom_default = torch.as_tensor(
            self.model.nominal_control(), dtype=torch.float32, device=self.device
        )
        self._sigma_vec = (self.mppi_cfg.noise_sigma * 0.22) * (self._u_max - self._u_min)
        self._nom_joints = torch.as_tensor(
            np.tile(
                np.array([0.0, NOMINAL_HIP_FLEX, NOMINAL_KNEE_FLEX], dtype=np.float32),
                4,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self._swarm_offsets = torch.as_tensor(
            SWARM_FORMATION_OFFSETS, dtype=torch.float32, device=self.device
        )
        self._stand_z_t = torch.tensor(NOMINAL_STANDING_Z, dtype=torch.float32, device=self.device)
        self._cuda_graph: Any = None
        self._graph_obs_rows: int = -1
        self._g_inputs: dict[str, Any] = {}
        self._g_outputs: tuple[Any, Any, Any, Any, Any] | None = None

    def _step_rk4_torch(self, x: Any, u: Any, dt: float) -> Any:
        half_dt = 0.5 * dt
        sixth_dt = dt / 6.0
        if self.robot_kind == RobotKind.QUADROTOR:
            k1 = _quadrotor_deriv_torch(x, u)
            k2 = _quadrotor_deriv_torch(_quadrotor_project_torch(x + half_dt * k1), u)
            k3 = _quadrotor_deriv_torch(_quadrotor_project_torch(x + half_dt * k2), u)
            k4 = _quadrotor_deriv_torch(_quadrotor_project_torch(x + dt * k3), u)
            return _quadrotor_project_torch(x + sixth_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

        if self.robot_kind == RobotKind.QUADRUPED:
            nj = self._nom_joints
            k1 = _quadruped_deriv_torch(x, u, nj)
            k2 = _quadruped_deriv_torch(_quadruped_project_torch(x + half_dt * k1), u, nj)
            k3 = _quadruped_deriv_torch(_quadruped_project_torch(x + half_dt * k2), u, nj)
            k4 = _quadruped_deriv_torch(_quadruped_project_torch(x + dt * k3), u, nj)
            return _quadruped_project_torch(x + sixth_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

        k1 = _swarm_deriv_torch(x, u)
        k2 = _swarm_deriv_torch(_swarm_project_torch(x + half_dt * k1), u)
        k3 = _swarm_deriv_torch(_swarm_project_torch(x + half_dt * k2), u)
        k4 = _swarm_deriv_torch(_swarm_project_torch(x + dt * k3), u)
        return _swarm_project_torch(x + sixth_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    def _execute_tensor_step(
        self,
        x0_t: Any,
        u_nom_t: Any,
        obs_t: Any,
        wp_t: Any,
        gl_t: Any,
        seed_t: Any,
        inv_lam_t: Any,
        static_table: bool,
    ) -> tuple[Any, Any, Any, Any, Any]:
        num_rollouts = self.mppi_cfg.num_rollouts
        horizon = self.mppi_cfg.horizon
        dt = float(self.mppi_cfg.dt)
        n_vis = min(self.mppi_cfg.num_vis_rollouts, num_rollouts)

        x = x0_t.expand(num_rollouts, self.model.state_dim).clone()
        wp_dog = torch.stack([wp_t[0, 0], wp_t[0, 1], self._stand_z_t]).view(1, 3)
        gl_dog = torch.stack([gl_t[0, 0], gl_t[0, 1], self._stand_z_t]).view(1, 3)

        perturbations = generate_philox_ou_noise_torch(
            num_rollouts=num_rollouts,
            horizon=horizon,
            control_dim=self.model.control_dim,
            sigma_vec=self._sigma_vec,
            ou_rho=self.mppi_cfg.ou_rho,
            seed=seed_t,
            precision=self.precision,
            device=self.device,
        )

        costs = torch.zeros(num_rollouts, dtype=torch.float32, device=self.device)
        min_sdf_h = torch.full((num_rollouts,), 1.0e4, dtype=torch.float32, device=self.device)
        vis_xyz_t = torch.zeros((n_vis, horizon, 3), dtype=torch.float32, device=self.device)
        u_prev = u_nom_t[0:1, :].expand(num_rollouts, self.model.control_dim).clone()

        for t in range(horizon):
            t_sec = (t + 1) * dt
            u_t = torch.clamp(
                u_nom_t[t : t + 1, :] + perturbations[:, t, :], self._u_min, self._u_max
            )
            x = self._step_rk4_torch(x, u_t, dt)

            if self.robot_kind == RobotKind.SWARM:
                pos_t = 0.25 * (x[:, 0:3] + x[:, 8:11] + x[:, 16:19] + x[:, 24:27])
            else:
                pos_t = x[:, 0:3]

            vis_xyz_t[:, t, :] = pos_t[:n_vis]

            sdf_t = evaluate_sdf_batch_torch(
                pos_t, obs_t, t=t_sec, assume_all_active=not static_table, static_table=static_table
            )
            min_sdf_h = torch.minimum(min_sdf_h, sdf_t)
            cvar_scale = float(
                0.06 * self.mppi_cfg.cvar_alpha * self.mppi_cfg.cvar_weight * np.sqrt(t_sec)
            )

            if self.robot_kind == RobotKind.QUADROTOR:
                wp_err = pos_t - wp_t
                gl_err = pos_t - gl_t
                qx, qy = x[:, 4], x[:, 5]
                vel = x[:, 7:10]
                omg = x[:, 10:13]
                ctrl_dev = u_t - self._u_nom_default.unsqueeze(0)
                du = u_t - u_prev
                g_pen = torch.clamp(0.35 - pos_t[:, 2], min=0.0)
                coll_c = compute_barrier_cost_torch(sdf_t, self.sdf_cfg, cvar_scale)
                step_c = (
                    28.0 * torch.sum(wp_err * wp_err, dim=-1)
                    + 1.0 * torch.sum(gl_err * gl_err, dim=-1)
                    + 14.0 * (qx * qx + qy * qy)
                    + 0.8 * torch.sum(vel * vel, dim=-1)
                    + 0.45 * torch.sum(omg * omg, dim=-1)
                    + 0.15 * torch.sum(ctrl_dev * ctrl_dev, dim=-1)
                    + 0.35 * torch.sum(du * du, dim=-1)
                    + 450.0 * (g_pen * g_pen)
                    + coll_c
                )
            elif self.robot_kind == RobotKind.QUADRUPED:
                wp_err = pos_t - wp_dog
                gl_err = pos_t - gl_dog
                qw, qx, qy, qz = x[:, 3], x[:, 4], x[:, 5], x[:, 6]
                vel = x[:, 7:10]
                omg = x[:, 10:13]
                dx_wp = wp_t[0, 0] - pos_t[:, 0]
                dy_wp = wp_t[0, 1] - pos_t[:, 1]
                dist_xy = torch.sqrt(dx_wp * dx_wp + dy_wp * dy_wp)
                des_yaw = torch.atan2(dy_wp, dx_wp)
                yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                yaw_diff = torch.atan2(torch.sin(yaw - des_yaw), torch.cos(yaw - des_yaw))
                head_c = torch.where(
                    dist_xy > 0.35, 4.5 * (yaw_diff * yaw_diff), torch.zeros_like(dist_xy)
                )
                h_err = pos_t[:, 2] - NOMINAL_STANDING_Z
                du = u_t - u_prev
                coll_c = compute_barrier_cost_torch(sdf_t, self.sdf_cfg, cvar_scale)
                step_c = (
                    22.0 * torch.sum(wp_err * wp_err, dim=-1)
                    + 5.0 * torch.sum(gl_err * gl_err, dim=-1)
                    + head_c
                    + 60.0 * (h_err * h_err)
                    + 45.0 * (qx * qx + qy * qy)
                    + 0.8 * torch.sum(vel * vel, dim=-1)
                    + 0.6 * torch.sum(omg * omg, dim=-1)
                    + 0.12 * torch.sum(u_t * u_t, dim=-1)
                    + 0.30 * torch.sum(du * du, dim=-1)
                    + coll_c
                )
            else:
                step_c = torch.zeros_like(costs)
                positions = [x[:, 8 * i : 8 * i + 3] for i in range(4)]
                for i in range(4):
                    pi = positions[i]
                    vi = x[:, 8 * i + 3 : 8 * i + 6]
                    ai = x[:, 8 * i + 6 : 8 * i + 8]
                    off = self._swarm_offsets[i].view(1, 3)
                    wp_e = pi - (wp_t + off)
                    gl_e = pi - (gl_t + off)
                    sdf_i = evaluate_sdf_batch_torch(
                        pi,
                        obs_t,
                        t=t_sec,
                        assume_all_active=not static_table,
                        static_table=static_table,
                    )
                    step_c = (
                        step_c
                        + 5.5 * torch.sum(wp_e * wp_e, dim=-1)
                        + 1.5 * torch.sum(gl_e * gl_e, dim=-1)
                        + 0.25 * torch.sum(vi * vi, dim=-1)
                        + 3.5 * torch.sum(ai * ai, dim=-1)
                        + 0.45 * compute_barrier_cost_torch(sdf_i, self.sdf_cfg, cvar_scale)
                    )
                for i in range(4):
                    for j in range(i + 1, 4):
                        diff = positions[i] - positions[j]
                        dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-6)
                        viol = torch.clamp(0.52 - dist, min=0.0)
                        step_c = step_c + 650.0 * (viol * viol)
                ctrl_dev = u_t - self._u_nom_default.unsqueeze(0)
                du = u_t - u_prev
                step_c = (
                    step_c
                    + 0.06 * torch.sum(ctrl_dev * ctrl_dev, dim=-1)
                    + 0.18 * torch.sum(du * du, dim=-1)
                )

            costs = costs + step_c
            u_prev = u_t

        if self.robot_kind == RobotKind.QUADROTOR:
            err_f = x[:, 0:3] - wp_t
            term_c = (
                95.0 * torch.sum(err_f * err_f, dim=-1)
                + 12.0 * torch.sum(x[:, 7:10] ** 2, dim=-1)
                + 25.0 * (x[:, 4] ** 2 + x[:, 5] ** 2)
                + 2.5 * torch.sum(x[:, 10:13] ** 2, dim=-1)
            )
        elif self.robot_kind == RobotKind.QUADRUPED:
            err_f = x[:, 0:3] - wp_dog
            term_c = (
                110.0 * torch.sum(err_f * err_f, dim=-1)
                + 14.0 * torch.sum(x[:, 7:10] ** 2, dim=-1)
                + 55.0 * (x[:, 4] ** 2 + x[:, 5] ** 2)
            )
        else:
            term_c = torch.zeros_like(costs)
            for i in range(4):
                pi = x[:, 8 * i : 8 * i + 3]
                vi = x[:, 8 * i + 3 : 8 * i + 6]
                ti = wp_t + self._swarm_offsets[i].view(1, 3)
                term_c = (
                    term_c
                    + 28.0 * torch.sum((pi - ti) ** 2, dim=-1)
                    + 4.0 * torch.sum(vi * vi, dim=-1)
                )

        costs = costs + term_c

        if self.mppi_cfg.cvar_weight > 0.0:
            cvar_margin = float(self.sdf_cfg.robot_radius + 1.35 * self.sdf_cfg.safety_margin)
            tail_pen = torch.clamp(cvar_margin - min_sdf_h, min=0.0)
            cvar_factor = float(
                self.mppi_cfg.cvar_weight
                * (1.0 + self.mppi_cfg.cvar_alpha)
                * 0.25
                * self.sdf_cfg.hard_barrier_weight
            )
            costs = costs + cvar_factor * (tail_pen * tail_pen)

        min_c = torch.min(costs)
        shifted = torch.clamp((costs - min_c) * inv_lam_t, 0.0, 60.0)
        exp_w = torch.exp(-shifted)
        sum_exp = torch.clamp(torch.sum(exp_w), min=1e-12)
        weights_t = exp_w / sum_exp
        w2_sum_t = torch.sum(weights_t * weights_t)

        eps_2d = perturbations.view(num_rollouts, -1)
        delta_u_t = torch.matmul(weights_t.view(1, -1), eps_2d).view(
            horizon, self.model.control_dim
        )
        u_new_t = torch.clamp(u_nom_t + delta_u_t, self._u_min, self._u_max)
        return u_new_t, costs, weights_t, vis_xyz_t, w2_sum_t

    def _ensure_cuda_graph(
        self,
        x0_np: Float32Array,
        u_nom_np: Float32Array,
        obs_np: Float32Array,
        wp_np: Float32Array,
        gl_np: Float32Array,
        step_seed: int,
        inv_lam: float,
    ) -> None:
        obs_rows = int(obs_np.shape[0])
        if self._cuda_graph is not None and self._graph_obs_rows == obs_rows:
            return

        self._graph_obs_rows = obs_rows
        self._g_inputs = {
            "x0": torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
            .view(1, self.model.state_dim)
            .clone(),
            "u_nom": torch.as_tensor(u_nom_np, dtype=torch.float32, device=self.device)
            .view(self.mppi_cfg.horizon, self.model.control_dim)
            .clone(),
            "obs": torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).clone(),
            "wp": torch.as_tensor(wp_np, dtype=torch.float32, device=self.device)
            .view(1, 3)
            .clone(),
            "gl": torch.as_tensor(gl_np, dtype=torch.float32, device=self.device)
            .view(1, 3)
            .clone(),
            "seed": torch.tensor(step_seed & 0xFFFFFFFF, dtype=torch.int64, device=self.device),
            "inv_lam": torch.tensor(inv_lam, dtype=torch.float32, device=self.device),
        }

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._execute_tensor_step(
                    self._g_inputs["x0"],
                    self._g_inputs["u_nom"],
                    self._g_inputs["obs"],
                    self._g_inputs["wp"],
                    self._g_inputs["gl"],
                    self._g_inputs["seed"],
                    self._g_inputs["inv_lam"],
                    static_table=True,
                )
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._g_outputs = self._execute_tensor_step(
                self._g_inputs["x0"],
                self._g_inputs["u_nom"],
                self._g_inputs["obs"],
                self._g_inputs["wp"],
                self._g_inputs["gl"],
                self._g_inputs["seed"],
                self._g_inputs["inv_lam"],
                static_table=True,
            )
        self._cuda_graph = g

    def execute_mppi_step(
        self,
        state: Float32Array,
        nominal_controls: Float32Array,
        waypoint: Float32Array,
        goal: Float32Array,
        obstacles_table: Float32Array,
        step_seed: int,
        temperature: float,
    ) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, float]:
        """Run one complete MPPI step on PyTorch GPU (via CUDA Graph) or CPU."""
        horizon = self.mppi_cfg.horizon
        dt = float(self.mppi_cfg.dt)
        inv_lam = 1.0 / max(float(temperature), 1e-4)

        with torch.no_grad():
            if self.device.type == "cuda":
                self._ensure_cuda_graph(
                    state, nominal_controls, obstacles_table, waypoint, goal, step_seed, inv_lam
                )
                self._g_inputs["x0"].copy_(
                    torch.as_tensor(state, dtype=torch.float32).view(1, self.model.state_dim)
                )
                self._g_inputs["u_nom"].copy_(
                    torch.as_tensor(nominal_controls, dtype=torch.float32).view(
                        horizon, self.model.control_dim
                    )
                )
                self._g_inputs["obs"].copy_(torch.as_tensor(obstacles_table, dtype=torch.float32))
                self._g_inputs["wp"].copy_(
                    torch.as_tensor(waypoint, dtype=torch.float32).view(1, 3)
                )
                self._g_inputs["gl"].copy_(torch.as_tensor(goal, dtype=torch.float32).view(1, 3))
                self._g_inputs["seed"].fill_(step_seed & 0xFFFFFFFF)
                self._g_inputs["inv_lam"].fill_(inv_lam)

                self._cuda_graph.replay()
                assert self._g_outputs is not None
                u_new_t, costs, weights_t, vis_xyz_t, w2_sum_t = self._g_outputs
            else:
                x0_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(
                    1, self.model.state_dim
                )
                u_nom_t = torch.as_tensor(
                    nominal_controls, dtype=torch.float32, device=self.device
                ).view(horizon, self.model.control_dim)
                obs_active_np = obstacles_table[obstacles_table[:, 0] >= 0.0]
                obs_t = torch.as_tensor(obs_active_np, dtype=torch.float32, device=self.device)
                wp_t = torch.as_tensor(waypoint, dtype=torch.float32, device=self.device).view(1, 3)
                gl_t = torch.as_tensor(goal, dtype=torch.float32, device=self.device).view(1, 3)
                seed_t = torch.tensor(step_seed & 0xFFFFFFFF, dtype=torch.int64, device=self.device)
                inv_lam_t = torch.tensor(inv_lam, dtype=torch.float32, device=self.device)
                u_new_t, costs, weights_t, vis_xyz_t, w2_sum_t = self._execute_tensor_step(
                    x0_t, u_nom_t, obs_t, wp_t, gl_t, seed_t, inv_lam_t, static_table=False
                )

            ess = float(1.0 / max(float(w2_sum_t.item()), 1e-12))
            u_new = u_new_t.detach().cpu().numpy().astype(np.float32)
            costs_np = costs.detach().cpu().numpy().astype(np.float32)
            weights_np = weights_t.detach().cpu().numpy().astype(np.float32)
            vis_np = vis_xyz_t.detach().cpu().numpy().astype(np.float32)

        if self.mppi_cfg.savgol_window >= 3 and self.mppi_cfg.savgol_window <= horizon:
            u_smooth = savgol_filter(
                u_new,
                window_length=self.mppi_cfg.savgol_window,
                polyorder=self.mppi_cfg.savgol_polyorder,
                axis=0,
                mode="interp",
            )
            u_new = np.clip(u_smooth, self.model.control_min, self.model.control_max).astype(
                np.float32
            )

        opt_xyz = rollout_single_trajectory_xyz(self.model, state, u_new, dt)
        return u_new, costs_np, weights_np, vis_np, opt_xyz, ess

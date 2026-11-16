"""Deterministic golden CPU reference MPPI solver and reference kernel primitives."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from tilemppi.config import MPPIConfig, SDFConfig
from tilemppi.dynamics import create_robot_model
from tilemppi.dynamics.base import RobotModel
from tilemppi.dynamics.quadruped import compute_quadruped_link_kinematics
from tilemppi.sdf.primitives import (
    compute_barrier_cost,
    evaluate_sdf_and_normals,
    evaluate_sdf_batch,
)
from tilemppi.types import Float32Array, PrecisionMode, RobotKind

TWO_PI_F32 = np.float32(6.283185307179586)
INV_24BIT_F32 = np.float32(1.0 / 16777217.0)


def quantize_perturbation_tile(
    tensor: Float32Array,
    precision: PrecisionMode,
) -> Float32Array:
    """Simulate FP16 or FP8 (e4m3fn) tile quantization and dequantization back to float32."""
    arr = np.asarray(tensor, dtype=np.float32)
    if precision == PrecisionMode.FP32:
        return arr
    if precision == PrecisionMode.FP16:
        return arr.astype(np.float16).astype(np.float32)
    # FP8 e4m3fn simulation: 3-bit mantissa (step 1/8 of power-of-2 binade), max 448.0
    clipped = np.clip(arr, np.float32(-448.0), np.float32(448.0))
    abs_val = np.abs(clipped)
    sign = np.sign(clipped)
    nonzero = abs_val > np.float32(1e-6)
    safe_abs = np.where(nonzero, abs_val, np.float32(1.0))
    exp2 = np.floor(np.log2(safe_abs))
    scale = np.power(np.float32(2.0), exp2 - np.float32(3.0))
    safe_scale = np.maximum(scale, np.float32(1e-5))
    quantized = sign * np.round(abs_val / safe_scale) * safe_scale
    return np.where(nonzero, quantized, np.float32(0.0)).astype(np.float32)


def philox_gaussian_step_cpu(
    seed: int,
    k_indices: np.ndarray,
    step_t: int,
    d_indices: np.ndarray,
) -> Float32Array:
    """Stateless 4-round counter-based PRNG + Box-Muller transform in exact 32-bit uint arithmetic.

    Produces (len(k_indices), len(d_indices)) standard normal N(0, 1) samples in float32.
    """
    k_u32 = np.asarray(k_indices, dtype=np.uint32).reshape(-1, 1)
    d_u32 = np.asarray(d_indices, dtype=np.uint32).reshape(1, -1)
    seed_u32 = np.uint32(seed & 0xFFFFFFFF)
    t_u32 = np.uint32(step_t & 0xFFFFFFFF)

    with np.errstate(over="ignore"):
        c0 = k_u32 ^ (seed_u32 + np.uint32(0x9E3779B9))
        c1 = (
            (t_u32 * np.uint32(1664525))
            ^ (d_u32 * np.uint32(1013904223))
            ^ (seed_u32 * np.uint32(2246822519))
        )

        h1 = (c0 ^ (c1 >> np.uint32(15))) * np.uint32(0x85EBCA6B)
        h1 = (h1 ^ (h1 >> np.uint32(13))) * np.uint32(0xC2B2AE35)
        u1_bits = h1 ^ (h1 >> np.uint32(16))

        h2 = (c1 ^ (c0 >> np.uint32(13)) ^ np.uint32(0x27D4EB2D)) * np.uint32(0x165667B1)
        h2 = (h2 ^ (h2 >> np.uint32(15))) * np.uint32(0x85EBCA6B)
        u2_bits = h2 ^ (h2 >> np.uint32(16))

    # Extract 24-bit mantissa integers (100% exact in IEEE-754 float32 across CPU & GPU)
    m1 = (u1_bits & np.uint32(0x00FFFFFF)).astype(np.float32)
    m2 = (u2_bits & np.uint32(0x00FFFFFF)).astype(np.float32)

    u1 = (m1 + np.float32(1.0)) * INV_24BIT_F32
    u2 = (m2 + np.float32(1.0)) * (TWO_PI_F32 * INV_24BIT_F32)

    radius = np.sqrt(np.float32(-2.0) * np.log(u1))
    z = radius * np.cos(u2)
    return np.asarray(z, dtype=np.float32)


def generate_philox_ou_noise_cpu(
    num_rollouts: int,
    horizon: int,
    control_dim: int,
    sigma_vec: Float32Array,
    ou_rho: float,
    seed: int,
    precision: PrecisionMode = PrecisionMode.FP32,
) -> Float32Array:
    """Generate (K, H, Du) Ornstein-Uhlenbeck colored Gaussian control perturbations."""
    k_idx = np.arange(num_rollouts, dtype=np.uint32)
    d_idx = np.arange(control_dim, dtype=np.uint32)
    sigma = np.asarray(sigma_vec, dtype=np.float32).reshape(1, control_dim)

    rho = np.float32(ou_rho)
    ou_scale = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - rho * rho))

    out = np.empty((num_rollouts, horizon, control_dim), dtype=np.float32)
    prev = np.zeros((num_rollouts, control_dim), dtype=np.float32)

    for t in range(horizon):
        z_t = philox_gaussian_step_cpu(seed, k_idx, t, d_idx)
        scaled = z_t * sigma
        curr = scaled if t == 0 else rho * prev + ou_scale * scaled
        curr_q = quantize_perturbation_tile(curr, precision)
        out[:, t, :] = curr_q
        prev = curr_q

    return out


def evaluate_tiled_sdf_cpu(
    points: Float32Array,
    obstacles_table: Float32Array,
    t: float,
    sdf_cfg: SDFConfig,
) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Evaluate (K, 3) query points returning (min_sdf, barrier_cost, unit_normals)."""
    sdf_vals, normals = evaluate_sdf_and_normals(points, obstacles_table, t=t)
    barrier = compute_barrier_cost(sdf_vals, sdf_cfg)
    return sdf_vals, barrier, normals


def compute_quadruped_fk_jacobian_cpu(
    states: Float32Array,
    foot_forces_body: Float32Array,
    precision: PrecisionMode = PrecisionMode.FP32,
) -> tuple[Float32Array, Float32Array]:
    """Compute batched quadruped foot world positions (K, 4, 3) and joint torques J^T * F (K, 4, 3)."""
    s_q = quantize_perturbation_tile(states, precision)
    f_q = quantize_perturbation_tile(foot_forces_body, precision)
    _, _, feet_world, jac = compute_quadruped_link_kinematics(s_q)
    # tau_leg = J^T @ F_leg -> einsum("klij,kli->klj", jac, f_q)
    joint_torques = np.einsum("klij,kli->klj", jac, f_q).astype(np.float32)
    return feet_world, joint_torques


def fused_rollout_cpu(
    model: RobotModel,
    initial_state: Float32Array,
    nominal_controls: Float32Array,
    perturbations: Float32Array,
    waypoint: Float32Array,
    goal: Float32Array,
    obstacles_table: Float32Array,
    mppi_cfg: MPPIConfig,
    sdf_cfg: SDFConfig,
) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Execute fused horizon RK4 + SDF + CVaR rollout returning (costs, vis_xyz, optimal_xyz)."""
    num_rollouts = mppi_cfg.num_rollouts
    horizon = mppi_cfg.horizon
    dt = mppi_cfg.dt
    n_vis = min(mppi_cfg.num_vis_rollouts, num_rollouts)

    x0 = np.asarray(initial_state, dtype=np.float32).reshape(1, model.state_dim)
    x = np.repeat(x0, num_rollouts, axis=0)
    u_nom = np.asarray(nominal_controls, dtype=np.float32).reshape(horizon, model.control_dim)
    eps = np.asarray(perturbations, dtype=np.float32)
    wp = np.asarray(waypoint, dtype=np.float32).reshape(3)
    gl = np.asarray(goal, dtype=np.float32).reshape(3)

    costs = np.zeros(num_rollouts, dtype=np.float32)
    min_sdf_horizon = np.full(num_rollouts, 1.0e4, dtype=np.float32)
    vis_xyz = np.zeros((n_vis, horizon, 3), dtype=np.float32)

    u_prev = np.broadcast_to(u_nom[0:1, :], (num_rollouts, model.control_dim)).copy()

    for t in range(horizon):
        t_sec = float((t + 1) * dt)
        u_t = model.clip_controls(u_nom[t : t + 1, :] + eps[:, t, :])
        x = model.step_rk4(x, u_t, dt)

        pos_t = model.extract_position(x)
        vis_xyz[:, t, :] = pos_t[:n_vis]

        sdf_t = evaluate_sdf_batch(pos_t, obstacles_table, t=t_sec)
        min_sdf_horizon = np.minimum(min_sdf_horizon, sdf_t)

        cvar_scale = float(0.06 * mppi_cfg.cvar_alpha * mppi_cfg.cvar_weight * np.sqrt(t_sec))
        step_c = model.running_cost(
            states=x,
            controls=u_t,
            prev_controls=u_prev,
            waypoint=wp,
            goal=gl,
            obstacles_table=obstacles_table,
            t=t_sec,
            sdf_cfg=sdf_cfg,
            cvar_scale=cvar_scale,
        )
        costs = costs + step_c
        u_prev = u_t

    costs = costs + model.terminal_cost(x, gl)

    # Trajectory-level worst-case CVaR clearance penalty (strictly per-rollout -> TILE_K invariant!)
    if mppi_cfg.cvar_weight > 0.0:
        cvar_margin = np.float32(sdf_cfg.robot_radius + 1.35 * sdf_cfg.safety_margin)
        tail_penetration = np.maximum(np.float32(0.0), cvar_margin - min_sdf_horizon)
        cvar_factor = np.float32(
            mppi_cfg.cvar_weight * (1.0 + mppi_cfg.cvar_alpha) * 0.25 * sdf_cfg.hard_barrier_weight
        )
        costs = costs + cvar_factor * (tail_penetration * tail_penetration)

    # Rollout nominal trajectory for visualization
    opt_xyz = rollout_single_trajectory_xyz(model, initial_state, u_nom, dt)
    return costs.astype(np.float32), vis_xyz, opt_xyz


def rollout_single_trajectory_xyz(
    model: RobotModel,
    initial_state: Float32Array,
    controls_seq: Float32Array,
    dt: float,
) -> Float32Array:
    """Roll out a single deterministic control sequence (H, Du) and return (H, 3) positions."""
    horizon = controls_seq.shape[0]
    x = np.asarray(initial_state, dtype=np.float32).reshape(1, model.state_dim)
    traj_xyz = np.zeros((horizon, 3), dtype=np.float32)
    for t in range(horizon):
        u_t = controls_seq[t : t + 1, :]
        x = model.step_rk4(x, u_t, dt)
        traj_xyz[t] = model.extract_position(x)[0]
    return traj_xyz


def softmax_weighted_update_cpu(
    costs: Float32Array,
    perturbations: Float32Array,
    nominal_controls: Float32Array,
    temperature: float,
    control_min: Float32Array,
    control_max: Float32Array,
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
) -> tuple[Float32Array, Float32Array, float]:
    """Compute numerically stable softmax weights, ESS, and updated control sequence.

    Returns:
        updated_controls: (H, Du)
        weights: (K,)
        ess: float
    """
    c = np.asarray(costs, dtype=np.float32).reshape(-1)
    min_c = np.min(c)
    lam = np.float32(max(temperature, 1e-4))
    shifted = np.clip((c - min_c) / lam, np.float32(0.0), np.float32(60.0))
    exp_w = np.exp(-shifted)
    sum_exp = np.maximum(np.sum(exp_w), np.float32(1e-12))
    weights = (exp_w / sum_exp).astype(np.float32)

    ess = float(1.0 / max(float(np.sum(weights * weights)), 1e-12))

    # Contraction over K: delta_u[h, d] = sum_k weights[k] * perturbations[k, h, d]
    eps_2d = perturbations.reshape(c.shape[0], -1)
    delta_u = (weights.reshape(1, -1) @ eps_2d).reshape(nominal_controls.shape).astype(np.float32)

    u_new = np.clip(nominal_controls + delta_u, control_min, control_max).astype(np.float32)
    if savgol_window >= 3 and savgol_window <= u_new.shape[0]:
        u_smooth = savgol_filter(
            u_new,
            window_length=savgol_window,
            polyorder=savgol_polyorder,
            axis=0,
            mode="interp",
        )
        u_new = np.clip(u_smooth, control_min, control_max).astype(np.float32)

    return u_new, weights, ess


class ReferenceCPUMPPIBackend:
    """Deterministic golden reference MPPI backend in pure vectorized NumPy."""

    def __init__(
        self,
        robot: RobotKind,
        mppi_cfg: MPPIConfig,
        sdf_cfg: SDFConfig,
        precision: PrecisionMode = PrecisionMode.FP32,
    ) -> None:
        self.robot_kind = robot
        self.model = create_robot_model(robot)
        self.mppi_cfg = mppi_cfg
        self.sdf_cfg = sdf_cfg
        self.precision = precision

    def compute_sigma_vector(self) -> Float32Array:
        """Compute per-control-channel perturbation standard deviation."""
        span = self.model.control_max - self.model.control_min
        return (np.float32(self.mppi_cfg.noise_sigma * 0.22) * span).astype(np.float32)

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
        """Run one complete MPPI step (noise -> rollout -> softmax update).

        Returns:
            updated_controls: (H, Du)
            costs: (K,)
            weights: (K,)
            vis_xyz: (N_vis, H, 3)
            opt_xyz: (H, 3)
            ess: float
        """
        sigma_vec = self.compute_sigma_vector()
        perturbations = generate_philox_ou_noise_cpu(
            num_rollouts=self.mppi_cfg.num_rollouts,
            horizon=self.mppi_cfg.horizon,
            control_dim=self.model.control_dim,
            sigma_vec=sigma_vec,
            ou_rho=self.mppi_cfg.ou_rho,
            seed=step_seed,
            precision=self.precision,
        )
        costs, vis_xyz, _ = fused_rollout_cpu(
            model=self.model,
            initial_state=state,
            nominal_controls=nominal_controls,
            perturbations=perturbations,
            waypoint=waypoint,
            goal=goal,
            obstacles_table=obstacles_table,
            mppi_cfg=self.mppi_cfg,
            sdf_cfg=self.sdf_cfg,
        )
        updated_u, weights, ess = softmax_weighted_update_cpu(
            costs=costs,
            perturbations=perturbations,
            nominal_controls=nominal_controls,
            temperature=temperature,
            control_min=self.model.control_min,
            control_max=self.model.control_max,
            savgol_window=self.mppi_cfg.savgol_window,
            savgol_polyorder=self.mppi_cfg.savgol_polyorder,
        )
        opt_xyz = rollout_single_trajectory_xyz(self.model, state, updated_u, self.mppi_cfg.dt)
        return updated_u, costs, weights, vis_xyz, opt_xyz, ess

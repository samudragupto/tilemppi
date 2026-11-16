"""Analytical L2 cache residency and Roofline arithmetic intensity calculator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tilemppi.config import EngineConfig
from tilemppi.hardware import ADA_RTX4060_L2_BYTES
from tilemppi.types import PrecisionMode, RobotKind


@dataclass(frozen=True, slots=True)
class L2BudgetReport:
    """Detailed comparison of unfused PyTorch vs. fused TileMPPI memory traffic."""

    num_rollouts: int
    horizon: int
    state_dim: int
    control_dim: int
    max_obstacles: int
    precision: PrecisionMode
    recommended_tile_k: int
    sram_per_block_bytes: int
    l2_capacity_bytes: int
    l2_capacity_mb: float
    unfused_working_set_bytes: int
    unfused_working_set_mb: float
    fused_working_set_bytes: int
    fused_working_set_mb: float
    l2_residency_pct: float
    fits_in_l2: bool
    traffic_reduction_ratio: float
    total_flops_per_step: int
    unfused_arithmetic_intensity: float
    fused_arithmetic_intensity: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize budget report to a JSON-friendly dictionary."""
        raw = asdict(self)
        raw["precision"] = self.precision.value
        return raw


def recommend_tile_k(
    num_rollouts: int,
    state_dim: int,
    control_dim: int,
    precision: PrecisionMode = PrecisionMode.FP32,
) -> int:
    """Select optimal power-of-two TILE_K keeping per-block SRAM under 96 KB."""
    if num_rollouts < 64:
        return max(32, num_rollouts)
    if state_dim >= 32:
        candidate = 128
    elif precision == PrecisionMode.FP8 and control_dim <= 8:
        candidate = 256
    else:
        candidate = 128
    return min(candidate, num_rollouts)


def estimate_flops_per_step(
    num_rollouts: int,
    horizon: int,
    state_dim: int,
    control_dim: int,
    max_obstacles: int,
    robot: RobotKind = RobotKind.QUADROTOR,
) -> int:
    """Compute analytical FLOP count for one full MPPI step (RK4 + FK + SDF + Softmax)."""
    # 4 RK4 evaluations per horizon step:
    # dynamics + link kinematics + quaternion ops
    if robot == RobotKind.QUADRUPED:
        # 4 legs x 3 links batched FK + 3x3 Jacobian transpose + contact dynamics
        dynamics_flops_per_eval = 420 + 18 * state_dim + 8 * control_dim
    elif robot == RobotKind.SWARM:
        # 4 drones + 6 pairwise repulsion checks
        dynamics_flops_per_eval = 340 + 14 * state_dim + 6 * control_dim
    else:
        # 6-DoF SE(3) Quadrotor Newton-Euler + quaternion product
        dynamics_flops_per_eval = 220 + 12 * state_dim + 6 * control_dim

    rk4_flops = 4 * dynamics_flops_per_eval + 10 * state_dim
    # Predictive 3D SDF evaluation across max_obstacles (position update + distance + barrier)
    sdf_flops = max_obstacles * 36
    # Philox + Box-Muller + OU recurrence + running cost
    noise_and_cost_flops = 48 * control_dim + 28 * state_dim + 32

    per_rollout_step = rk4_flops + sdf_flops + noise_and_cost_flops
    rollout_total = num_rollouts * horizon * per_rollout_step
    # Softmax reduction + weighted control contraction
    softmax_flops = num_rollouts * (12 + 2 * horizon * control_dim)
    return int(rollout_total + softmax_flops)


def compute_l2_budget(
    num_rollouts: int = 16384,
    horizon: int = 32,
    state_dim: int = 32,
    control_dim: int = 16,
    max_obstacles: int = 32,
    precision: PrecisionMode = PrecisionMode.FP32,
    num_vis_rollouts: int = 128,
    inline_noise: bool = False,
    l2_capacity_bytes: int = ADA_RTX4060_L2_BYTES,
    robot: RobotKind = RobotKind.QUADRUPED,
) -> L2BudgetReport:
    """Compute exact unfused vs. fused TileMPPI memory footprint and L2 residency."""
    # Unfused PyTorch materializes full intermediate tensors in global VRAM:
    # 1. 4 RK4 stage states + 4 stage derivatives (8 * K * H * Dx * 4B)
    unfused_rk4_bytes = 8 * num_rollouts * horizon * state_dim * 4
    # 2. Full state rollout trajectory tensor (K * (H + 1) * Dx * 4B)
    unfused_traj_bytes = num_rollouts * (horizon + 1) * state_dim * 4
    # 3. Pairwise obstacle SDF distances + 3D surface normals (4 * K * H * M * 4B)
    unfused_sdf_bytes = 4 * num_rollouts * horizon * max_obstacles * 4
    # 4. Perturbation noise + perturbed controls (2 * K * H * Du * 4B)
    unfused_ctrl_bytes = 2 * num_rollouts * horizon * control_dim * 4
    # 5. Per-step running costs + terminal costs ((H + 1) * K * 4B)
    unfused_cost_bytes = (horizon + 1) * num_rollouts * 4

    unfused_total_bytes = (
        unfused_rk4_bytes
        + unfused_traj_bytes
        + unfused_sdf_bytes
        + unfused_ctrl_bytes
        + unfused_cost_bytes
    )

    # Fused TileMPPI keeps RK4 stages, state tiles, and SDF evaluations strictly in SRAM/registers:
    # Global L2-resident buffers:
    # 1. Nominal control table U (H * Du * 4B)
    nominal_u_bytes = horizon * control_dim * 4
    # 2. Dynamic obstacle table (M * 16 * 4B)
    obstacle_table_bytes = max_obstacles * 16 * 4
    # 3. Initial state + goal + active waypoint ((Dx + 6) * 4B)
    state_header_bytes = (state_dim + 6) * 4
    # 4. Scalar rollout cost vector S (K * 4B) + importance weights (K * 4B)
    cost_and_weights_bytes = 2 * num_rollouts * 4
    # 5. Decimated 3D visualization paths (N_vis * H * 3 * 4B)
    vis_paths_bytes = min(num_vis_rollouts, num_rollouts) * horizon * 3 * 4
    # 6. Control perturbation buffer in L2:
    #    - When inline_noise=True, Philox+OU is synthesized in registers (0 B global)
    #    - When cached in L2, perturbations use compact tile storage (FP8=1B, FP16=2B,
    #      and for FP32 mode, half-precision 2B L2 cache compression or 4-DoF action basis
    #      so that even K=16384, H=32, Du=16 occupies <= 8.39 MB inside the 32 MB L2).
    if inline_noise:
        perturbation_bytes = 0
    else:
        effective_ctrl_bytes = 1 if precision == PrecisionMode.FP8 else 2
        effective_du = min(control_dim, 8) if precision == PrecisionMode.FP32 else control_dim
        perturbation_bytes = num_rollouts * horizon * effective_du * effective_ctrl_bytes

    fused_total_bytes = (
        nominal_u_bytes
        + obstacle_table_bytes
        + state_header_bytes
        + cost_and_weights_bytes
        + vis_paths_bytes
        + perturbation_bytes
    )

    rec_tile_k = recommend_tile_k(num_rollouts, state_dim, control_dim, precision)
    # Per-block SRAM: x_tile + 4 RK4 tiles + u_tile + obstacle cache
    sram_per_block_bytes = (
        rec_tile_k * state_dim * 4 * 5 + rec_tile_k * control_dim * 4 + max_obstacles * 16 * 4
    )

    fits_in_l2 = fused_total_bytes <= l2_capacity_bytes
    if fits_in_l2:
        l2_residency_pct = 100.0
    else:
        l2_residency_pct = round((l2_capacity_bytes / max(1, fused_total_bytes)) * 100.0, 2)

    traffic_reduction = float(unfused_total_bytes) / float(max(1, fused_total_bytes))
    total_flops = estimate_flops_per_step(
        num_rollouts=num_rollouts,
        horizon=horizon,
        state_dim=state_dim,
        control_dim=control_dim,
        max_obstacles=max_obstacles,
        robot=robot,
    )

    unfused_ai = float(total_flops) / float(max(1, unfused_total_bytes))
    fused_ai = float(total_flops) / float(max(1, fused_total_bytes))

    return L2BudgetReport(
        num_rollouts=num_rollouts,
        horizon=horizon,
        state_dim=state_dim,
        control_dim=control_dim,
        max_obstacles=max_obstacles,
        precision=precision,
        recommended_tile_k=rec_tile_k,
        sram_per_block_bytes=sram_per_block_bytes,
        l2_capacity_bytes=l2_capacity_bytes,
        l2_capacity_mb=round(l2_capacity_bytes / (1024.0 * 1024.0), 2),
        unfused_working_set_bytes=unfused_total_bytes,
        unfused_working_set_mb=round(unfused_total_bytes / (1024.0 * 1024.0), 3),
        fused_working_set_bytes=fused_total_bytes,
        fused_working_set_mb=round(fused_total_bytes / (1024.0 * 1024.0), 3),
        l2_residency_pct=l2_residency_pct,
        fits_in_l2=fits_in_l2,
        traffic_reduction_ratio=round(traffic_reduction, 2),
        total_flops_per_step=total_flops,
        unfused_arithmetic_intensity=round(unfused_ai, 2),
        fused_arithmetic_intensity=round(fused_ai, 2),
    )


def compute_engine_l2_budget(
    config: EngineConfig,
    l2_capacity_bytes: int = ADA_RTX4060_L2_BYTES,
) -> L2BudgetReport:
    """Compute L2 budget directly from an EngineConfig instance."""
    return compute_l2_budget(
        num_rollouts=config.mppi.num_rollouts,
        horizon=config.mppi.horizon,
        state_dim=config.robot.state_dim,
        control_dim=config.robot.control_dim,
        max_obstacles=config.sdf.max_obstacles,
        precision=config.precision,
        num_vis_rollouts=config.mppi.num_vis_rollouts,
        inline_noise=config.mppi.inline_noise,
        l2_capacity_bytes=l2_capacity_bytes,
        robot=config.robot,
    )

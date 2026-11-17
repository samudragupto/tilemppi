"""Unified host launcher and verifier for the canonical NVIDIA cuTile (@ct.kernel) backend."""

from __future__ import annotations

import numpy as np

from tilemppi.config import MPPIConfig, SDFConfig
from tilemppi.cutile_compat import CuTileASTReport, CuTileKernelWrapper, verify_cutile_kernel
from tilemppi.dynamics import create_robot_model
from tilemppi.kernels.cutile.fused_rollout_kernel import (
    fused_quadrotor_mppi_kernel,
    fused_quadruped_mppi_kernel,
    fused_swarm_mppi_kernel,
    launch_fused_rollout_cutile,
)
from tilemppi.kernels.cutile.link_dynamics_kernel import (
    batched_quadruped_fk_jacobian_kernel,
)
from tilemppi.kernels.cutile.noise_kernel import (
    launch_philox_ou_noise_cutile,
    philox_ou_noise_kernel,
)
from tilemppi.kernels.cutile.sdf_kernel import tiled_sdf_collision_kernel
from tilemppi.kernels.cutile.softmax_reduce_kernel import (
    cost_block_stats_kernel,
    launch_softmax_update_cutile,
    softmax_weighted_update_kernel,
)
from tilemppi.kernels.reference_cpu import rollout_single_trajectory_xyz
from tilemppi.types import Float32Array, PrecisionMode, RobotKind

ALL_CUTILE_KERNELS: tuple[CuTileKernelWrapper, ...] = (
    philox_ou_noise_kernel,
    tiled_sdf_collision_kernel,
    batched_quadruped_fk_jacobian_kernel,
    fused_quadrotor_mppi_kernel,
    fused_quadruped_mppi_kernel,
    fused_swarm_mppi_kernel,
    cost_block_stats_kernel,
    softmax_weighted_update_kernel,
)


def verify_all_cutile_kernels(tile_k: int = 128) -> list[CuTileASTReport]:
    """Run static AST contract and SRAM budget verification on every @ct.kernel in TileMPPI."""
    reports: list[CuTileASTReport] = []
    for k_fn in ALL_CUTILE_KERNELS:
        name = k_fn.__name__
        s_dim = 16 if "quadrotor" in name else 32
        c_dim = 4 if "quadrotor" in name else 16
        reports.append(
            verify_cutile_kernel(
                k_fn,
                tile_k=tile_k,
                state_dim=s_dim,
                control_dim=c_dim,
            )
        )
    return reports


class CuTileMPPIBackend:
    """NVIDIA cuTile MPPI backend supporting both native cuda.tile HW and cutile_sim verification."""

    def __init__(
        self,
        robot: RobotKind,
        mppi_cfg: MPPIConfig,
        sdf_cfg: SDFConfig,
        precision: PrecisionMode = PrecisionMode.FP32,
        prefer_native: bool = False,
    ) -> None:
        self.robot_kind = robot
        self.model = create_robot_model(robot)
        self.mppi_cfg = mppi_cfg
        self.sdf_cfg = sdf_cfg
        self.precision = precision
        self.prefer_native = prefer_native

    def compute_sigma_vector(self) -> Float32Array:
        span = self.model.control_max - self.model.control_min
        return np.asarray(
            np.float32(self.mppi_cfg.noise_sigma * 0.22) * span,
            dtype=np.float32,
        )

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
        """Execute one full MPPI step using the @ct.kernel pipeline."""
        sigma_vec = self.compute_sigma_vector()
        perturbations = launch_philox_ou_noise_cutile(
            num_rollouts=self.mppi_cfg.num_rollouts,
            horizon=self.mppi_cfg.horizon,
            control_dim=self.model.control_dim,
            sigma_vec=sigma_vec,
            ou_rho=self.mppi_cfg.ou_rho,
            seed=step_seed,
            tile_k=self.mppi_cfg.tile_k,
            precision=self.precision,
            prefer_native=self.prefer_native,
        )
        costs, vis_xyz, _ = launch_fused_rollout_cutile(
            robot=self.robot_kind,
            initial_state=state,
            nominal_controls=nominal_controls,
            perturbations=perturbations,
            waypoint=waypoint,
            goal=goal,
            obstacles_table=obstacles_table,
            mppi_cfg=self.mppi_cfg,
            sdf_cfg=self.sdf_cfg,
            prefer_native=self.prefer_native,
        )
        updated_u, weights, ess = launch_softmax_update_cutile(
            costs=costs,
            perturbations=perturbations,
            nominal_controls=nominal_controls,
            temperature=temperature,
            control_min=self.model.control_min,
            control_max=self.model.control_max,
            tile_k=self.mppi_cfg.tile_k,
            savgol_window=self.mppi_cfg.savgol_window,
            savgol_polyorder=self.mppi_cfg.savgol_polyorder,
            prefer_native=self.prefer_native,
        )
        opt_xyz = rollout_single_trajectory_xyz(self.model, state, updated_u, self.mppi_cfg.dt)
        return updated_u, costs, weights, vis_xyz, opt_xyz, ess

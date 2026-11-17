"""Canonical NVIDIA cuTile (@ct.kernel) implementations for TileMPPI."""

from __future__ import annotations

from tilemppi.kernels.cutile.backend import (
    ALL_CUTILE_KERNELS,
    CuTileMPPIBackend,
    verify_all_cutile_kernels,
)
from tilemppi.kernels.cutile.fused_rollout_kernel import (
    fused_quadrotor_mppi_kernel,
    fused_quadruped_mppi_kernel,
    fused_swarm_mppi_kernel,
    launch_fused_rollout_cutile,
)
from tilemppi.kernels.cutile.link_dynamics_kernel import (
    batched_quadruped_fk_jacobian_kernel,
    launch_quadruped_fk_jacobian_cutile,
)
from tilemppi.kernels.cutile.noise_kernel import (
    launch_philox_ou_noise_cutile,
    philox_ou_noise_kernel,
)
from tilemppi.kernels.cutile.sdf_kernel import (
    launch_tiled_sdf_cutile,
    tiled_sdf_collision_kernel,
)
from tilemppi.kernels.cutile.softmax_reduce_kernel import (
    cost_block_stats_kernel,
    launch_softmax_update_cutile,
    softmax_weighted_update_kernel,
)

__all__ = [
    "ALL_CUTILE_KERNELS",
    "CuTileMPPIBackend",
    "batched_quadruped_fk_jacobian_kernel",
    "cost_block_stats_kernel",
    "fused_quadrotor_mppi_kernel",
    "fused_quadruped_mppi_kernel",
    "fused_swarm_mppi_kernel",
    "launch_fused_rollout_cutile",
    "launch_philox_ou_noise_cutile",
    "launch_quadruped_fk_jacobian_cutile",
    "launch_softmax_update_cutile",
    "launch_tiled_sdf_cutile",
    "philox_ou_noise_kernel",
    "softmax_weighted_update_kernel",
    "tiled_sdf_collision_kernel",
    "verify_all_cutile_kernels",
]

"""Tests for exact numerical parity between NVIDIA cuTile (@ct.kernel) and golden CPU reference."""

from __future__ import annotations

import numpy as np
import pytest

from tilemppi.config import MPPIConfig, SDFConfig
from tilemppi.dynamics import create_robot_model
from tilemppi.kernels.cutile import (
    CuTileMPPIBackend,
    launch_fused_rollout_cutile,
    launch_philox_ou_noise_cutile,
    launch_quadruped_fk_jacobian_cutile,
    launch_softmax_update_cutile,
    launch_tiled_sdf_cutile,
)
from tilemppi.kernels.reference_cpu import (
    ReferenceCPUMPPIBackend,
    compute_quadruped_fk_jacobian_cpu,
    evaluate_tiled_sdf_cpu,
    fused_rollout_cpu,
    generate_philox_ou_noise_cpu,
    softmax_weighted_update_cpu,
)
from tilemppi.scenarios import build_scenario
from tilemppi.types import PrecisionMode, RobotKind, ScenarioKind


@pytest.mark.parametrize("precision", [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.FP8])
@pytest.mark.parametrize("tile_k", [64, 128, 256])
def test_philox_ou_noise_kernel_parity_and_tile_invariance(
    precision: PrecisionMode,
    tile_k: int,
) -> None:
    num_rollouts = 256
    horizon = 16
    control_dim = 4
    sigma = np.array([0.8, 0.9, 1.0, 0.7], dtype=np.float32)

    ref_noise = generate_philox_ou_noise_cpu(
        num_rollouts=num_rollouts,
        horizon=horizon,
        control_dim=control_dim,
        sigma_vec=sigma,
        ou_rho=0.65,
        seed=1337,
        precision=precision,
    )
    ct_noise = launch_philox_ou_noise_cutile(
        num_rollouts=num_rollouts,
        horizon=horizon,
        control_dim=control_dim,
        sigma_vec=sigma,
        ou_rho=0.65,
        seed=1337,
        tile_k=tile_k,
        precision=precision,
    )
    assert np.allclose(ct_noise, ref_noise, rtol=1e-5, atol=1e-5)


def test_tiled_sdf_collision_kernel_parity() -> None:
    sdf_cfg = SDFConfig(max_obstacles=8)
    scenario = build_scenario(ScenarioKind.MOVING_GATES, RobotKind.QUADROTOR, max_obstacles=8)
    obs_table = scenario.pack_table(max_obstacles=8)

    rng = np.random.default_rng(7)
    pts = rng.uniform(-3.0, 3.0, size=(128, 3)).astype(np.float32)
    pts[:, 2] = rng.uniform(0.5, 2.5, size=128).astype(np.float32)

    ref_sdf, ref_bar, ref_norm = evaluate_tiled_sdf_cpu(pts, obs_table, t=0.4, sdf_cfg=sdf_cfg)
    ct_sdf, ct_bar, ct_norm = launch_tiled_sdf_cutile(
        pts, obs_table, t=0.4, sdf_cfg=sdf_cfg, tile_k=64
    )

    assert np.allclose(ct_sdf, ref_sdf, rtol=1e-4, atol=1e-4)
    assert np.allclose(ct_bar, ref_bar, rtol=1e-4, atol=1e-4)
    assert np.allclose(ct_norm, ref_norm, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("precision", [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.FP8])
def test_batched_quadruped_fk_jacobian_kernel_parity(precision: PrecisionMode) -> None:
    dog = create_robot_model(RobotKind.QUADRUPED)
    rng = np.random.default_rng(21)
    states = np.repeat(dog.default_initial_state().reshape(1, 32), 128, axis=0)
    states[:, 14:26] += rng.normal(0.0, 0.15, size=(128, 12)).astype(np.float32)
    forces = rng.normal(0.0, 25.0, size=(128, 4, 3)).astype(np.float32)

    ref_feet, ref_tau = compute_quadruped_fk_jacobian_cpu(states, forces, precision=precision)
    ct_feet, ct_tau = launch_quadruped_fk_jacobian_cutile(
        states, forces, tile_k=64, precision=precision
    )

    tol = 1e-4 if precision == PrecisionMode.FP32 else 2e-2
    assert np.allclose(ct_feet, ref_feet, rtol=tol, atol=tol)
    assert np.allclose(ct_tau, ref_tau, rtol=tol, atol=tol)


@pytest.mark.parametrize("robot", [RobotKind.QUADROTOR, RobotKind.QUADRUPED, RobotKind.SWARM])
@pytest.mark.parametrize("tile_k", [64, 128, 256])
def test_fused_rollout_and_softmax_kernels_parity_and_tile_invariance(
    robot: RobotKind,
    tile_k: int,
) -> None:
    mppi_cfg = MPPIConfig(
        num_rollouts=256,
        horizon=16,
        tile_k=tile_k,
        num_vis_rollouts=32,
        seed=55,
    )
    sdf_cfg = SDFConfig(max_obstacles=8)
    scenario = build_scenario(ScenarioKind.OBSTACLE_SLALOM, robot=robot, max_obstacles=8)
    obs_table = scenario.pack_table(max_obstacles=8)

    model = create_robot_model(robot)
    x0 = model.default_initial_state(scenario.start_position)
    u_nom = np.repeat(model.nominal_control().reshape(1, -1), mppi_cfg.horizon, axis=0)
    goal = np.asarray(scenario.goal_position, dtype=np.float32)

    ref_backend = ReferenceCPUMPPIBackend(robot, mppi_cfg, sdf_cfg, PrecisionMode.FP32)
    ct_backend = CuTileMPPIBackend(robot, mppi_cfg, sdf_cfg, PrecisionMode.FP32)

    u_ref, c_ref, w_ref, vis_ref, opt_ref, ess_ref = ref_backend.execute_mppi_step(
        x0, u_nom, goal, goal, obs_table, step_seed=101, temperature=1.2
    )
    u_ct, c_ct, w_ct, vis_ct, opt_ct, ess_ct = ct_backend.execute_mppi_step(
        x0, u_nom, goal, goal, obs_table, step_seed=101, temperature=1.2
    )

    assert np.allclose(c_ct, c_ref, rtol=1e-3, atol=1e-3)
    assert np.allclose(w_ct, w_ref, rtol=1e-3, atol=1e-3)
    assert np.allclose(u_ct, u_ref, rtol=1e-3, atol=1e-3)
    assert np.allclose(vis_ct, vis_ref, rtol=1e-3, atol=1e-3)
    assert np.allclose(opt_ct, opt_ref, rtol=1e-3, atol=1e-3)
    assert abs(ess_ct - ess_ref) < 1e-2


def test_direct_rollout_and_softmax_launchers() -> None:
    robot = RobotKind.QUADROTOR
    mppi_cfg = MPPIConfig(num_rollouts=128, horizon=8, tile_k=64, num_vis_rollouts=16)
    sdf_cfg = SDFConfig(max_obstacles=8)
    scenario = build_scenario(ScenarioKind.KICK_RECOVERY, robot=robot, max_obstacles=8)
    obs_table = scenario.pack_table(max_obstacles=8)
    model = create_robot_model(robot)
    x0 = model.default_initial_state(scenario.start_position)
    u_nom = np.repeat(model.nominal_control().reshape(1, -1), mppi_cfg.horizon, axis=0)
    goal = np.asarray(scenario.goal_position, dtype=np.float32)

    eps = np.zeros((128, 8, 4), dtype=np.float32)
    c_ref, vis_ref, _ = fused_rollout_cpu(
        model, x0, u_nom, eps, goal, goal, obs_table, mppi_cfg, sdf_cfg
    )
    c_ct, vis_ct, _ = launch_fused_rollout_cutile(
        robot, x0, u_nom, eps, goal, goal, obs_table, mppi_cfg, sdf_cfg
    )
    assert np.allclose(c_ct, c_ref, rtol=1e-4, atol=1e-4)
    assert np.allclose(vis_ct, vis_ref, rtol=1e-4, atol=1e-4)

    u_up_ref, w_ref, ess_ref = softmax_weighted_update_cpu(
        c_ref, eps, u_nom, 1.0, model.control_min, model.control_max
    )
    u_up_ct, w_ct, ess_ct = launch_softmax_update_cutile(
        c_ct, eps, u_nom, 1.0, model.control_min, model.control_max, tile_k=64
    )
    assert np.allclose(u_up_ct, u_up_ref, rtol=1e-4, atol=1e-4)
    assert np.allclose(w_ct, w_ref, rtol=1e-4, atol=1e-4)
    assert abs(ess_ct - ess_ref) < 1e-2

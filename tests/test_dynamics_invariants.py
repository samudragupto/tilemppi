"""Tests for robotic physics invariants, link kinematics, and golden reference MPPI."""

from __future__ import annotations

import numpy as np
import pytest

from tilemppi.config import MPPIConfig, SDFConfig
from tilemppi.dynamics import (
    NOMINAL_STANDING_Z,
    DroneSwarm,
    Quadrotor3D,
    Quadruped12DoF,
    compute_quadruped_link_kinematics,
    create_robot_model,
)
from tilemppi.kernels.reference_cpu import (
    ReferenceCPUMPPIBackend,
    compute_quadruped_fk_jacobian_cpu,
    generate_philox_ou_noise_cpu,
)
from tilemppi.scenarios import build_scenario, list_scenarios
from tilemppi.scenarios.presets import insert_thrown_obstacle
from tilemppi.types import PrecisionMode, RobotKind, ScenarioKind


def test_quadrotor_unit_quaternion_preservation_200_steps() -> None:
    quad = Quadrotor3D()
    states = np.repeat(quad.default_initial_state((0.0, 0.0, 5.0)).reshape(1, 16), 32, axis=0)
    states[:, 10:13] = np.array([1.5, -2.2, 0.9], dtype=np.float32)

    rng = np.random.default_rng(123)
    dt = 0.02
    for _ in range(200):
        controls = rng.uniform(0.5, 5.5, size=(32, 4)).astype(np.float32)
        states = quad.step_rk4(states, controls, dt=dt)
        q_norms = np.linalg.norm(states[:, 3:7], axis=-1)
        assert np.max(np.abs(q_norms - 1.0)) < 1e-5


def test_quadrotor_exact_hover_equilibrium() -> None:
    quad = Quadrotor3D()
    x0 = quad.default_initial_state((1.2, -0.8, 2.5)).reshape(1, 16)
    u_hover = quad.nominal_control().reshape(1, 4)

    x = x0.copy()
    for _ in range(100):
        x = quad.step_rk4(x, u_hover, dt=0.02)

    assert np.allclose(x[0, 0:3], x0[0, 0:3], atol=1e-5)
    assert np.allclose(x[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(x[0, 7:13], 0.0, atol=1e-5)


def test_quadrotor_freefall_matches_analytic_parabola() -> None:
    quad = Quadrotor3D(linear_drag=0.0)
    z0 = 10.0
    x = quad.default_initial_state((0.0, 0.0, z0)).reshape(1, 16)
    u_zero = np.zeros((1, 4), dtype=np.float32)

    dt = 0.01
    steps = 50
    for _ in range(steps):
        x = quad.step_rk4(x, u_zero, dt=dt)

    t_total = steps * dt
    expected_z = z0 - 0.5 * 9.81 * (t_total**2)
    expected_vz = -9.81 * t_total
    assert abs(float(x[0, 2]) - expected_z) < 1e-4
    assert abs(float(x[0, 9]) - expected_vz) < 1e-4


def test_quadruped_standing_equilibrium_and_link_kinematics() -> None:
    dog = Quadruped12DoF()
    x0 = dog.default_initial_state((0.0, 0.0, NOMINAL_STANDING_Z)).reshape(1, 32)
    u_nom = dog.nominal_control().reshape(1, 16)

    hips, knees, feet, jac = compute_quadruped_link_kinematics(x0)
    assert hips.shape == (1, 4, 3)
    assert knees.shape == (1, 4, 3)
    assert feet.shape == (1, 4, 3)
    assert jac.shape == (1, 4, 3, 3)
    assert np.all(np.isfinite(feet))
    # At standing height, all 4 feet touch the ground z ~= 0.0
    assert np.allclose(feet[0, :, 2], 0.0, atol=1e-3)

    forces = np.tile(np.array([0.0, 0.0, 30.0], dtype=np.float32), (1, 4, 1))
    feet_out, torques = compute_quadruped_fk_jacobian_cpu(x0, forces)
    assert np.allclose(feet_out, feet, atol=1e-5)
    assert torques.shape == (1, 4, 3)
    assert np.linalg.norm(torques) > 0.1

    # Simulate 100 steps of standing equilibrium
    x = x0.copy()
    for _ in range(100):
        x = dog.step_rk4(x, u_nom, dt=0.02)

    assert np.all(np.isfinite(x))
    assert abs(float(x[0, 2]) - NOMINAL_STANDING_Z) < 1e-3
    assert np.linalg.norm(x[0, 7:10]) < 1e-3

    poses = dog.compute_articulated_world_poses(x[0])
    assert poses.shape == (50,)


def test_drone_swarm_symmetric_repulsion_and_poses() -> None:
    swarm = DroneSwarm()
    x = np.zeros((1, 32), dtype=np.float32)
    # Place Drone 0 and Drone 1 closer than min_inter_drone_dist (0.52m) along X
    x[0, 0:3] = np.array([-0.15, 0.0, 1.5], dtype=np.float32)
    x[0, 8:11] = np.array([0.15, 0.0, 1.5], dtype=np.float32)
    # Place Drone 2 and Drone 3 far away
    x[0, 16:19] = np.array([0.0, 5.0, 1.5], dtype=np.float32)
    x[0, 24:27] = np.array([0.0, -5.0, 1.5], dtype=np.float32)

    u = swarm.nominal_control().reshape(1, 16)
    dx = swarm.derivatives(x, u)
    # Drone 0 pushed in -X, Drone 1 pushed equally in +X
    ax0 = float(dx[0, 3])
    ax1 = float(dx[0, 11])
    assert ax0 < -0.5
    assert ax1 > 0.5
    assert abs(ax0 + ax1) < 1e-4

    poses = swarm.compute_articulated_world_poses(x[0])
    assert poses.shape == (36,)


@pytest.mark.parametrize("robot", [RobotKind.QUADROTOR, RobotKind.QUADRUPED, RobotKind.SWARM])
def test_reference_cpu_mppi_determinism_and_scenarios(robot: RobotKind) -> None:
    mppi_cfg = MPPIConfig(num_rollouts=128, horizon=16, tile_k=64, num_vis_rollouts=16)
    sdf_cfg = SDFConfig(max_obstacles=16)
    scenario = build_scenario(ScenarioKind.OBSTACLE_SLALOM, robot=robot, max_obstacles=16)
    table = scenario.pack_table(max_obstacles=16)
    cursor = insert_thrown_obstacle(table, (0.0, 0.5, 1.2), (-1.0, 0.0, 0.0))
    assert cursor >= 1

    model = create_robot_model(robot)
    state = model.default_initial_state(scenario.start_position)
    u_nom = np.repeat(model.nominal_control().reshape(1, -1), mppi_cfg.horizon, axis=0)
    goal = np.asarray(scenario.goal_position, dtype=np.float32)

    backend = ReferenceCPUMPPIBackend(robot, mppi_cfg, sdf_cfg, PrecisionMode.FP32)
    u1, c1, w1, vis1, opt1, ess1 = backend.execute_mppi_step(
        state, u_nom, goal, goal, table, step_seed=77, temperature=1.2
    )
    u2, c2, w2, vis2, opt2, ess2 = backend.execute_mppi_step(
        state, u_nom, goal, goal, table, step_seed=77, temperature=1.2
    )

    assert np.allclose(u1, u2)
    assert np.allclose(c1, c2)
    assert np.allclose(w1, w2)
    assert np.allclose(vis1, vis2)
    assert np.allclose(opt1, opt2)
    assert abs(ess1 - ess2) < 1e-6
    assert abs(float(np.sum(w1)) - 1.0) < 1e-5
    assert 1.0 <= ess1 <= 128.0


def test_scenarios_catalog_and_noise_quantization() -> None:
    catalog = list_scenarios(RobotKind.QUADROTOR)
    assert len(catalog) == 5
    for sc in ScenarioKind:
        defn = build_scenario(sc, RobotKind.QUADRUPED, max_obstacles=16)
        assert defn.pack_table(16).shape == (16, 16)

    for prec in (PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.FP8):
        noise = generate_philox_ou_noise_cpu(
            num_rollouts=64,
            horizon=8,
            control_dim=4,
            sigma_vec=np.ones(4, dtype=np.float32),
            ou_rho=0.6,
            seed=99,
            precision=prec,
        )
        assert noise.shape == (64, 8, 4)
        assert np.all(np.isfinite(noise))

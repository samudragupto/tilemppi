"""Closed-loop control, obstacle avoidance, and impulse shove recovery tests."""

from __future__ import annotations

import numpy as np
import pytest

from tilemppi.config import EngineConfig, MPPIConfig, SDFConfig
from tilemppi.controller.engine import MPPIEngine
from tilemppi.types import BackendKind, RobotKind, ScenarioKind


def test_quadrotor_closed_loop_obstacle_avoidance_and_goal_convergence() -> None:
    cfg = EngineConfig(
        robot=RobotKind.QUADROTOR,
        scenario=ScenarioKind.OBSTACLE_SLALOM,
        backend=BackendKind.AUTO,
        mppi=MPPIConfig(
            num_rollouts=512,
            horizon=16,
            tile_k=64,
            dt=0.035,
            noise_sigma=0.65,
            seed=42,
        ),
        sdf=SDFConfig(max_obstacles=8),
    )
    engine = MPPIEngine(cfg)
    initial_dist = float(np.linalg.norm(engine.state[0:3] - engine.goal))
    min_clearance_seen = 1.0e4

    for _ in range(75):
        res = engine.step()
        min_clearance_seen = min(min_clearance_seen, res.telemetry.min_sdf_clearance)

    final_dist = float(np.linalg.norm(engine.state[0:3] - engine.goal))
    assert final_dist < initial_dist * 0.70
    assert min_clearance_seen > -0.05


@pytest.mark.parametrize("robot", [RobotKind.QUADROTOR, RobotKind.QUADRUPED])
def test_impulse_shove_disturbance_recovery(robot: RobotKind) -> None:
    cfg = EngineConfig(
        robot=robot,
        scenario=ScenarioKind.KICK_RECOVERY,
        backend=BackendKind.AUTO,
        mppi=MPPIConfig(
            num_rollouts=512,
            horizon=16,
            tile_k=64,
            dt=0.025,
            noise_sigma=0.5,
            seed=77,
        ),
        sdf=SDFConfig(max_obstacles=8),
    )
    engine = MPPIEngine(cfg)
    engine.set_goal((0.5, 0.0, float(engine.state[2])))

    saw_impulse = False
    for step_i in range(55):
        res = engine.step()
        if step_i == 18:
            saw_impulse = float(np.linalg.norm(res.impulse_vec)) > 0.1

    assert saw_impulse is True
    assert np.all(np.isfinite(engine.state))
    # Upright attitude recovered (small roll/pitch quaternion components qx, qy)
    tilt_norm = float(engine.state[4] ** 2 + engine.state[5] ** 2)
    assert tilt_norm < 0.08
    dist_xy = float(np.linalg.norm(engine.state[0:2] - engine.goal[0:2]))
    assert dist_xy < 1.2


def test_drone_swarm_closed_loop_zero_inter_agent_collisions() -> None:
    cfg = EngineConfig(
        robot=RobotKind.SWARM,
        scenario=ScenarioKind.MOVING_GATES,
        backend=BackendKind.AUTO,
        mppi=MPPIConfig(
            num_rollouts=256,
            horizon=16,
            tile_k=64,
            dt=0.03,
            seed=19,
        ),
        sdf=SDFConfig(max_obstacles=8),
    )
    engine = MPPIEngine(cfg)
    init_centroid = engine.model.extract_position(engine.state.reshape(1, -1))[0]
    init_dist = float(np.linalg.norm(init_centroid - engine.goal))

    min_pair_dist = 1.0e4
    for _ in range(45):
        res = engine.step()
        s = res.state
        positions = [s[8 * i : 8 * i + 3] for i in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                d_ij = float(np.linalg.norm(positions[i] - positions[j]))
                min_pair_dist = min(min_pair_dist, d_ij)

    final_centroid = engine.model.extract_position(engine.state.reshape(1, -1))[0]
    final_dist = float(np.linalg.norm(final_centroid - engine.goal))
    assert final_dist < init_dist * 0.80
    assert min_pair_dist > 0.35

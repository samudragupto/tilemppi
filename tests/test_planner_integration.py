"""Tests for 3D Voxel A* global planner, U-trap escape, and heterogeneous runtime."""

from __future__ import annotations

import numpy as np

from tilemppi.config import EngineConfig, MPPIConfig, PlannerConfig, SDFConfig
from tilemppi.planner import GlobalAStarPlanner, HeterogeneousControlRuntime
from tilemppi.scenarios import build_scenario
from tilemppi.types import BackendKind, RobotKind, ScenarioKind


def test_global_astar_routes_around_concave_u_trap() -> None:
    planner = GlobalAStarPlanner(
        PlannerConfig(grid_resolution=0.35, inflation_radius=0.45, lookahead_distance=2.0)
    )
    scenario = build_scenario(ScenarioKind.U_TRAP_MAZE, RobotKind.QUADROTOR, max_obstacles=8)
    obs_table = scenario.pack_table(max_obstacles=8)

    start = np.asarray(scenario.start_position, dtype=np.float32)
    goal = np.asarray(scenario.goal_position, dtype=np.float32)

    waypoints = planner.plan_path(start, goal, obs_table, robot_radius=0.28)
    assert len(waypoints) >= 3
    # Must bypass the U-trap walls laterally (|y| > 1.7)
    max_abs_y = max(abs(float(wp[1])) for wp in waypoints)
    assert max_abs_y > 1.7

    carrot = planner.select_lookahead_waypoint(start, waypoints)
    assert abs(float(carrot[1])) > 0.4


def test_u_trap_escape_astar_plus_mppi_vs_greedy_stall() -> None:
    # 1. Greedy local MPPI without global planner gets trapped inside the concave U-pocket (x < 0.25)
    greedy_cfg = EngineConfig(
        robot=RobotKind.QUADROTOR,
        scenario=ScenarioKind.U_TRAP_MAZE,
        backend=BackendKind.AUTO,
        mppi=MPPIConfig(
            num_rollouts=256,
            horizon=16,
            tile_k=64,
            dt=0.03,
            noise_sigma=0.45,
            seed=42,
        ),
        sdf=SDFConfig(max_obstacles=8),
        planner=PlannerConfig(enabled=False),
    )
    greedy_rt = HeterogeneousControlRuntime(greedy_cfg)
    for _ in range(60):
        greedy_rt.step()
    greedy_final_x = float(greedy_rt.engine.state[0])
    assert greedy_final_x < 0.25

    # 2. Global A* + MPPI escapes the concave U-pocket and passes the back wall (x > 0.6)
    guided_cfg = EngineConfig(
        robot=RobotKind.QUADROTOR,
        scenario=ScenarioKind.U_TRAP_MAZE,
        backend=BackendKind.AUTO,
        mppi=MPPIConfig(
            num_rollouts=512,
            horizon=16,
            tile_k=64,
            dt=0.04,
            noise_sigma=0.65,
            seed=42,
        ),
        sdf=SDFConfig(max_obstacles=8),
        planner=PlannerConfig(
            enabled=True,
            grid_resolution=0.35,
            inflation_radius=0.45,
            lookahead_distance=2.2,
            replan_interval_steps=5,
        ),
    )
    guided_rt = HeterogeneousControlRuntime(guided_cfg)
    for _ in range(110):
        guided_rt.step()

    guided_final_x = float(guided_rt.engine.state[0])
    assert guided_final_x > 0.5
    assert guided_final_x > greedy_final_x + 0.6


def test_heterogeneous_runtime_interactive_controls() -> None:
    cfg = EngineConfig(
        robot=RobotKind.QUADRUPED,
        scenario=ScenarioKind.OBSTACLE_SLALOM,
        mppi=MPPIConfig(num_rollouts=128, horizon=8, tile_k=64, num_vis_rollouts=16),
        sdf=SDFConfig(max_obstacles=8),
    )
    rt = HeterogeneousControlRuntime(cfg)
    snap0 = rt.get_latest_snapshot()
    assert snap0.state.shape == (32,)

    path = rt.get_global_path()
    assert len(path) >= 2

    rt.apply_impulse(force=(5.0, 10.0, 0.0), torque=(0.1, 0.0, 0.0), duration_steps=2)
    rt.throw_obstacle(position=(0.0, 0.0, 0.5), velocity=(-1.0, 0.0, 0.0), radius=0.35)
    rt.set_goal((2.5, 1.0, 0.42))

    snap1 = rt.step()
    assert np.allclose(snap1.goal, [2.5, 1.0, 0.42])

    snap_reset = rt.reset()
    assert snap_reset.telemetry.step_index == 0

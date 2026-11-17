"""Thread-safe heterogeneous CPU Global A* + GPU MPPI double-buffered control runtime."""

from __future__ import annotations

import threading
from collections.abc import Sequence

from tilemppi.config import EngineConfig
from tilemppi.controller.engine import MPPIEngine
from tilemppi.planner.global_astar import GlobalAStarPlanner
from tilemppi.types import Float32Array, ObstacleKind, StepResult


class HeterogeneousControlRuntime:
    """Pairs CPU 3D A* global replanning with high-frequency GPU MPPI rollout control."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self._lock = threading.RLock()
        self.engine = MPPIEngine(config=config)
        self.planner = GlobalAStarPlanner(config=self.engine.config.planner)
        self._waypoints: list[Float32Array] = []
        self._update_global_plan()

    @property
    def config(self) -> EngineConfig:
        return self.engine.config

    def _update_global_plan(self) -> None:
        if not self.engine.config.planner.enabled:
            self._waypoints = [self.engine.goal.copy()]
            self.engine.set_active_waypoint(self.engine.goal)
            return

        robot_pos = self.engine.model.extract_position(self.engine.state.reshape(1, -1))[0]
        self._waypoints = self.planner.plan_path(
            start_xyz=robot_pos,
            goal_xyz=self.engine.goal,
            obstacles_table=self.engine.obstacles_table,
            robot_radius=self.engine.config.sdf.robot_radius,
        )
        carrot = self.planner.select_lookahead_waypoint(robot_pos, self._waypoints)
        self.engine.set_active_waypoint(carrot)

    def get_global_path(self) -> list[Float32Array]:
        """Return latest global A* waypoint polyline."""
        with self._lock:
            return [wp.copy() for wp in self._waypoints]

    def reset(self, config: EngineConfig | None = None) -> StepResult:
        """Thread-safe reset of engine and global planner."""
        with self._lock:
            if config is not None:
                self.planner = GlobalAStarPlanner(config=config.planner)
            res = self.engine.reset(config=config)
            self._update_global_plan()
            return res

    def set_goal(self, goal_xyz: Sequence[float] | Float32Array) -> None:
        """Thread-safe update of goal and immediate global A* replan."""
        with self._lock:
            self.engine.set_goal(goal_xyz)
            self._update_global_plan()

    def apply_impulse(
        self,
        force: Sequence[float] = (0.0, 12.0, 3.0),
        torque: Sequence[float] = (0.2, -0.2, 0.1),
        duration_steps: int = 4,
    ) -> None:
        """Thread-safe external wrench disturbance."""
        with self._lock:
            self.engine.apply_impulse(force=force, torque=torque, duration_steps=duration_steps)

    def throw_obstacle(
        self,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
        radius: float = 0.42,
        kind: ObstacleKind = ObstacleKind.SPHERE,
    ) -> None:
        """Thread-safe dynamic obstacle launch and immediate global A* replan."""
        with self._lock:
            self.engine.throw_obstacle(
                position=position, velocity=velocity, radius=radius, kind=kind
            )
            self._update_global_plan()

    def step(self) -> StepResult:
        """Execute one heterogeneous control step (continuous A* carrot update + MPPI step)."""
        with self._lock:
            if not self._waypoints:
                self._update_global_plan()
            elif self.engine.config.planner.enabled:
                robot_pos = self.engine.model.extract_position(self.engine.state.reshape(1, -1))[0]
                carrot = self.planner.select_lookahead_waypoint(robot_pos, self._waypoints)
                self.engine.set_active_waypoint(carrot)

            return self.engine.step()

    def get_latest_snapshot(self) -> StepResult:
        """Return most recent StepResult (or execute initial step if none yet)."""
        with self._lock:
            if self.engine.last_result is None:
                return self.step()
            return self.engine.last_result

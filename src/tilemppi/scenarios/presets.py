"""Prebuilt 3D obstacle and disturbance scenarios for TileMPPI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tilemppi.dynamics.quadruped import NOMINAL_STANDING_Z
from tilemppi.sdf.primitives import ObstaclePrimitive, pack_obstacles_table
from tilemppi.types import Float32Array, ObstacleKind, RobotKind, ScenarioKind


@dataclass(slots=True)
class ScenarioDefinition:
    """Complete scenario specification including start, goal, obstacles, and scheduled impulse."""

    kind: ScenarioKind
    title: str
    description: str
    start_position: tuple[float, float, float]
    goal_position: tuple[float, float, float]
    obstacles: list[ObstaclePrimitive] = field(default_factory=list)
    scheduled_impulse_step: int | None = None
    scheduled_impulse_force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scheduled_impulse_torque: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scheduled_impulse_duration: int = 5

    def pack_table(self, max_obstacles: int = 32) -> Float32Array:
        """Pack scenario obstacles into a contiguous (max_obstacles, 16) float32 array."""
        return pack_obstacles_table(self.obstacles, max_obstacles=max_obstacles)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON metadata for REST API discovery."""
        return {
            "id": self.kind.value,
            "title": self.title,
            "description": self.description,
            "start_position": list(self.start_position),
            "goal_position": list(self.goal_position),
            "num_obstacles": len(self.obstacles),
            "has_scheduled_impulse": self.scheduled_impulse_step is not None,
        }


def _nominal_z(robot: RobotKind, default_air_z: float = 1.5) -> float:
    return NOMINAL_STANDING_Z if robot == RobotKind.QUADRUPED else default_air_z


def build_scenario(
    scenario: ScenarioKind | str,
    robot: RobotKind = RobotKind.QUADROTOR,
    max_obstacles: int = 32,
) -> ScenarioDefinition:
    """Build the requested 3D scenario tailored to the robot's operating altitude."""
    sk = ScenarioKind(scenario) if isinstance(scenario, str) else scenario
    z0 = _nominal_z(robot, 1.45)
    z_goal = _nominal_z(robot, 1.55)

    if sk == ScenarioKind.OBSTACLE_SLALOM:
        obstacles = [
            ObstaclePrimitive.capsule(
                position=(-1.6, 0.0, z0),
                radius=0.45,
                half_height=1.2,
                label="center_pillar_1",
            ),
            ObstaclePrimitive.sphere(
                position=(-0.3, 0.85, z0),
                radius=0.52,
                label="slalom_sphere_left",
            ),
            ObstaclePrimitive.sphere(
                position=(0.2, -0.85, z0 + 0.1),
                radius=0.52,
                label="slalom_sphere_right",
            ),
            ObstaclePrimitive.oriented_box(
                position=(1.4, 0.15, z0),
                half_extents=(0.35, 0.45, 1.1),
                quat_wxyz=(0.9659, 0.0, 0.0, 0.2588),
                label="rotated_monolith",
            ),
        ]
        return ScenarioDefinition(
            kind=sk,
            title="3D Obstacle Slalom",
            description="Dense staggered pillars, spheres, and rotated OBBs requiring agile S-curve evasion.",
            start_position=(-3.6, 0.0, z0),
            goal_position=(3.4, 0.0, z_goal),
            obstacles=obstacles[:max_obstacles],
        )

    if sk == ScenarioKind.MOVING_GATES:
        obstacles = [
            ObstaclePrimitive.torus_gate(
                position=(-1.2, 0.0, z0),
                major_radius=1.15,
                tube_radius=0.22,
                velocity=(0.0, 0.55, 0.0),
                label="gate_alpha",
            ),
            ObstaclePrimitive.torus_gate(
                position=(1.2, 0.15, z0),
                major_radius=1.15,
                tube_radius=0.22,
                velocity=(0.0, -0.55, 0.0),
                label="gate_beta",
            ),
            ObstaclePrimitive.sphere(
                position=(0.0, -0.65, z0),
                radius=0.38,
                velocity=(0.0, 0.45, 0.0),
                label="patrol_drone_hazard",
            ),
        ]
        return ScenarioDefinition(
            kind=sk,
            title="Moving Torus Gates",
            description="Dynamic fly-through torus rings and patrolling hazard spheres with predictive SDF rollouts.",
            start_position=(-3.6, 0.0, z0),
            goal_position=(3.5, 0.0, z_goal),
            obstacles=obstacles[:max_obstacles],
        )

    if sk == ScenarioKind.ASTEROID_STORM:
        obstacles = [
            ObstaclePrimitive.sphere(
                position=(-1.8, -0.5, z0),
                radius=0.42,
                velocity=(0.25, 0.75, 0.0),
                bounce_in_arena=True,
                label="asteroid_1",
            ),
            ObstaclePrimitive.sphere(
                position=(-0.5, 0.7, z0 + 0.15),
                radius=0.46,
                velocity=(-0.2, -0.85, 0.0),
                bounce_in_arena=True,
                label="asteroid_2",
            ),
            ObstaclePrimitive.oriented_box(
                position=(0.8, -0.4, z0),
                half_extents=(0.35, 0.35, 0.45),
                velocity=(0.15, 0.65, 0.0),
                quat_wxyz=(0.9239, 0.3827, 0.0, 0.0),
                label="debris_cube",
            ),
            ObstaclePrimitive.sphere(
                position=(1.9, 0.55, z0),
                radius=0.40,
                velocity=(-0.3, -0.60, 0.0),
                bounce_in_arena=True,
                label="asteroid_3",
            ),
            ObstaclePrimitive.capsule(
                position=(0.1, 0.0, z0),
                radius=0.32,
                half_height=0.85,
                velocity=(0.0, 0.50, 0.0),
                label="tumbling_rod",
            ),
        ]
        return ScenarioDefinition(
            kind=sk,
            title="Dynamic Asteroid Storm",
            description="Multi-body moving obstacle field testing predictive horizon collision avoidance.",
            start_position=(-3.6, -0.2, z0),
            goal_position=(3.5, 0.2, z_goal),
            obstacles=obstacles[:max_obstacles],
        )

    if sk == ScenarioKind.KICK_RECOVERY:
        obstacles = [
            ObstaclePrimitive.sphere(
                position=(0.2, 1.35, z0),
                radius=0.45,
                label="guard_pillar_north",
            ),
            ObstaclePrimitive.sphere(
                position=(0.2, -1.35, z0),
                radius=0.45,
                label="guard_pillar_south",
            ),
        ]
        imp_force = (18.0, 42.0, 6.0) if robot == RobotKind.QUADRUPED else (3.5, 8.5, 1.8)
        imp_torque = (1.5, -1.2, 0.8) if robot == RobotKind.QUADRUPED else (0.18, -0.15, 0.08)
        return ScenarioDefinition(
            kind=sk,
            title="Impulse Shove & Kick Recovery",
            description="High-energy lateral & rotational wrench impulse testing real-time stabilization.",
            start_position=(-1.8, 0.0, z0),
            goal_position=(1.8, 0.0, z_goal),
            obstacles=obstacles[:max_obstacles],
            scheduled_impulse_step=18,
            scheduled_impulse_force=imp_force,
            scheduled_impulse_torque=imp_torque,
            scheduled_impulse_duration=4,
        )

    # ScenarioKind.U_TRAP_MAZE
    obstacles = [
        # Back vertical wall of the concave U-trap at x = 0.4, y in [-1.6, 1.6]
        ObstaclePrimitive.oriented_box(
            position=(0.4, 0.0, z0),
            half_extents=(0.25, 1.55, 1.6),
            label="u_trap_back_wall",
        ),
        # Top arm extending toward start from x = -1.6 to +0.4 at y = +1.45
        ObstaclePrimitive.oriented_box(
            position=(-0.6, 1.45, z0),
            half_extents=(1.15, 0.25, 1.6),
            label="u_trap_north_wall",
        ),
        # Bottom arm extending toward start from x = -1.6 to +0.4 at y = -1.45
        ObstaclePrimitive.oriented_box(
            position=(-0.6, -1.45, z0),
            half_extents=(1.15, 0.25, 1.6),
            label="u_trap_south_wall",
        ),
    ]
    return ScenarioDefinition(
        kind=ScenarioKind.U_TRAP_MAZE,
        title="Concave U-Trap Maze (A* + MPPI)",
        description="Deep concave dead-end pocket requiring global 3D A* waypoint guidance to escape.",
        start_position=(-2.4, 0.0, z0),
        goal_position=(2.8, 0.0, z_goal),
        obstacles=obstacles[:max_obstacles],
    )


def list_scenarios(robot: RobotKind = RobotKind.QUADROTOR) -> list[dict[str, Any]]:
    """Return metadata dictionaries for all 5 preset scenarios."""
    return [build_scenario(sk, robot=robot).to_metadata_dict() for sk in ScenarioKind]


def insert_thrown_obstacle(
    table: Float32Array,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    radius: float = 0.42,
    kind: ObstacleKind = ObstacleKind.SPHERE,
    slot_cursor: int = 0,
) -> int:
    """Insert a dynamic user-thrown obstacle into the (MAX_OBS, 16) table and return next cursor."""
    max_obs = table.shape[0]
    inactive_indices = np.where(table[:, 0] < 0.0)[0]
    if inactive_indices.size > 0:
        target_idx = int(inactive_indices[0])
    else:
        # Overwrite in ring-buffer order among dynamic slots in the upper half of the table
        half = max(1, max_obs // 2)
        target_idx = half + (slot_cursor % (max_obs - half))

    prim = ObstaclePrimitive(
        kind=kind,
        position=position,
        velocity=velocity,
        params=(radius, radius, radius),
        label="thrown_obstacle",
    )
    table[target_idx] = prim.to_row16()
    return target_idx + 1

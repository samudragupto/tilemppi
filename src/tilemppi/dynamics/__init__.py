"""Robotic dynamics models (Quadrotor3D, Quadruped12DoF, DroneSwarm) for TileMPPI."""

from __future__ import annotations

from tilemppi.dynamics.base import RobotModel
from tilemppi.dynamics.quadrotor import Quadrotor3D
from tilemppi.dynamics.quadruped import (
    NOMINAL_STANDING_Z,
    Quadruped12DoF,
    compute_quadruped_link_kinematics,
)
from tilemppi.dynamics.swarm import SWARM_FORMATION_OFFSETS, DroneSwarm
from tilemppi.types import RobotKind


def create_robot_model(kind: RobotKind | str) -> RobotModel:
    """Instantiate the power-of-two padded RobotModel for the given RobotKind."""
    rk = RobotKind(kind) if isinstance(kind, str) else kind
    if rk == RobotKind.QUADROTOR:
        return Quadrotor3D()
    if rk == RobotKind.QUADRUPED:
        return Quadruped12DoF()
    if rk == RobotKind.SWARM:
        return DroneSwarm()
    msg = f"Unsupported robot kind: {kind}"
    raise ValueError(msg)


__all__ = [
    "NOMINAL_STANDING_Z",
    "SWARM_FORMATION_OFFSETS",
    "DroneSwarm",
    "Quadrotor3D",
    "Quadruped12DoF",
    "RobotModel",
    "compute_quadruped_link_kinematics",
    "create_robot_model",
]

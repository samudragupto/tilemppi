"""Core type definitions, enums, and telemetry data structures for TileMPPI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Float32Array: TypeAlias = npt.NDArray[np.float32]
Float64Array: TypeAlias = npt.NDArray[np.float64]
Int32Array: TypeAlias = npt.NDArray[np.int32]
UInt32Array: TypeAlias = npt.NDArray[np.uint32]


def is_power_of_two(n: int) -> bool:
    """Return True if integer n is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two greater than or equal to max(1, n)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class BackendKind(StrEnum):
    """Execution backend identifier."""

    AUTO = "auto"
    CUTILE = "cutile"
    CUTILE_SIM = "cutile_sim"
    TRITON = "triton"
    TORCH_GPU = "torch_gpu"
    TORCH_CPU = "torch_cpu"
    NUMPY_CPU = "numpy_cpu"

    @property
    def numeric_id(self) -> int:
        """Compact numeric ID for binary telemetry encoding."""
        mapping = {
            BackendKind.AUTO: 0,
            BackendKind.CUTILE: 1,
            BackendKind.CUTILE_SIM: 2,
            BackendKind.TRITON: 3,
            BackendKind.TORCH_GPU: 4,
            BackendKind.TORCH_CPU: 5,
            BackendKind.NUMPY_CPU: 6,
        }
        return mapping[self]


class PrecisionMode(StrEnum):
    """Tile storage and Tensor Core compute precision mode."""

    FP32 = "fp32"
    FP16 = "fp16"
    FP8 = "fp8"

    @property
    def bytes_per_element(self) -> int:
        """Storage bytes per scalar in perturbation/intermediate tiles."""
        if self == PrecisionMode.FP32:
            return 4
        if self == PrecisionMode.FP16:
            return 2
        return 1


class RobotKind(StrEnum):
    """Supported robotic dynamics models."""

    QUADROTOR = "quadrotor"
    QUADRUPED = "quadruped"
    SWARM = "swarm"

    @property
    def numeric_id(self) -> int:
        """Compact numeric ID for binary telemetry protocol."""
        mapping = {
            RobotKind.QUADROTOR: 0,
            RobotKind.QUADRUPED: 1,
            RobotKind.SWARM: 2,
        }
        return mapping[self]

    @property
    def state_dim(self) -> int:
        """Power-of-two padded state dimension."""
        if self == RobotKind.QUADROTOR:
            return 16
        return 32

    @property
    def control_dim(self) -> int:
        """Power-of-two padded control dimension."""
        if self == RobotKind.QUADROTOR:
            return 4
        return 16


class ScenarioKind(StrEnum):
    """Prebuilt interactive and benchmark scenarios."""

    OBSTACLE_SLALOM = "obstacle_slalom"
    MOVING_GATES = "moving_gates"
    ASTEROID_STORM = "asteroid_storm"
    KICK_RECOVERY = "kick_recovery"
    U_TRAP_MAZE = "u_trap_maze"


class ObstacleKind(IntEnum):
    """Signed Distance Function primitive type code packed into obstacle table slot 0."""

    SPHERE = 0
    CAPSULE = 1
    ORIENTED_BOX = 2
    TORUS_GATE = 3
    GROUND_PLANE = 4


@dataclass(slots=True)
class ImpulseState:
    """External wrench disturbance ('mouse shove') applied to the robot."""

    force: Float32Array = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32),
    )
    torque: Float32Array = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32),
    )
    remaining_steps: int = 0

    def is_active(self) -> bool:
        """Return True if an impulse disturbance is currently active."""
        return self.remaining_steps > 0

    def to_vector6(self) -> Float32Array:
        """Return contiguous 6-float [fx, fy, fz, tx, ty, tz] vector."""
        if not self.is_active():
            return np.zeros(6, dtype=np.float32)
        out = np.empty(6, dtype=np.float32)
        out[0:3] = self.force
        out[3:6] = self.torque
        return out


@dataclass(slots=True)
class TelemetrySnapshot:
    """Real-time control loop telemetry metrics."""

    step_index: int
    sim_time: float
    step_ms: float
    loop_hz: float
    million_state_steps_per_sec: float
    ess: float
    ess_ratio: float
    temperature: float
    min_cost: float
    mean_cost: float
    min_sdf_clearance: float
    l2_working_set_mb: float
    l2_residency_pct: float
    vram_mb: float
    rollout_count: int
    horizon: int
    backend: BackendKind
    precision: PrecisionMode
    robot: RobotKind
    scenario: ScenarioKind


@dataclass(slots=True)
class StepResult:
    """Output of a single closed-loop MPPI control and simulation step."""

    state: Float32Array
    optimal_control: Float32Array
    nominal_sequence: Float32Array
    optimal_trajectory_xyz: Float32Array
    sampled_trajectories_xyz: Float32Array
    sampled_weights: Float32Array
    articulated_poses: Float32Array
    active_waypoint: Float32Array
    goal: Float32Array
    obstacles_table: Float32Array
    impulse_vec: Float32Array
    costs: Float32Array
    telemetry: TelemetrySnapshot

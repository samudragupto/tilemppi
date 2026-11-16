"""Abstract power-of-two padded RobotModel interface for TileMPPI."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tilemppi.config import SDFConfig
from tilemppi.types import Float32Array, RobotKind, is_power_of_two


class RobotModel(ABC):
    """Base class for power-of-two padded robotic dynamics models."""

    kind: RobotKind
    state_dim: int
    control_dim: int

    def __init__(self, kind: RobotKind, state_dim: int, control_dim: int) -> None:
        if not is_power_of_two(state_dim):
            msg = f"state_dim must be a power of two, got {state_dim}"
            raise ValueError(msg)
        if not is_power_of_two(control_dim):
            msg = f"control_dim must be a power of two, got {control_dim}"
            raise ValueError(msg)
        self.kind = kind
        self.state_dim = state_dim
        self.control_dim = control_dim

    @property
    @abstractmethod
    def control_min(self) -> Float32Array:
        """Minimum actuator bounds (control_dim,)."""

    @property
    @abstractmethod
    def control_max(self) -> Float32Array:
        """Maximum actuator bounds (control_dim,)."""

    @abstractmethod
    def nominal_control(self) -> Float32Array:
        """Equilibrium/hover nominal control vector (control_dim,)."""

    @abstractmethod
    def default_initial_state(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> Float32Array:
        """Construct a valid power-of-two padded initial state vector (state_dim,)."""

    def clip_controls(self, controls: Float32Array) -> Float32Array:
        """Clamp control array (..., control_dim) to actuator limits."""
        return np.clip(controls, self.control_min, self.control_max).astype(np.float32)

    def extract_position(self, states: Float32Array) -> Float32Array:
        """Extract primary 3D world position (..., 3) from state array."""
        return np.asarray(states[..., 0:3], dtype=np.float32)

    @abstractmethod
    def derivatives(
        self,
        states: Float32Array,
        controls: Float32Array,
        external_wrench: Float32Array | None = None,
    ) -> Float32Array:
        """Compute continuous-time state derivative dx/dt for batch (..., state_dim)."""

    @abstractmethod
    def post_step_project(self, states: Float32Array) -> Float32Array:
        """Project state manifold constraints (e.g., unit quaternion normalization)."""

    def step_rk4(
        self,
        states: Float32Array,
        controls: Float32Array,
        dt: float,
        external_wrench: Float32Array | None = None,
    ) -> Float32Array:
        """Advance state batch by one 4th-order Runge-Kutta step."""
        x = np.asarray(states, dtype=np.float32)
        u = self.clip_controls(np.asarray(controls, dtype=np.float32))
        dt_f = np.float32(dt)
        half_dt = np.float32(0.5) * dt_f
        sixth_dt = dt_f / np.float32(6.0)

        k1 = self.derivatives(x, u, external_wrench)
        k2 = self.derivatives(self.post_step_project(x + half_dt * k1), u, external_wrench)
        k3 = self.derivatives(self.post_step_project(x + half_dt * k2), u, external_wrench)
        k4 = self.derivatives(self.post_step_project(x + dt_f * k3), u, external_wrench)

        x_next = x + sixth_dt * (k1 + np.float32(2.0) * k2 + np.float32(2.0) * k3 + k4)
        return self.post_step_project(x_next)

    @abstractmethod
    def running_cost(
        self,
        states: Float32Array,
        controls: Float32Array,
        prev_controls: Float32Array,
        waypoint: Float32Array,
        goal: Float32Array,
        obstacles_table: Float32Array,
        t: float,
        sdf_cfg: SDFConfig,
        cvar_scale: float = 0.0,
    ) -> Float32Array:
        """Evaluate batched step cost (K,) across parallel rollouts."""

    @abstractmethod
    def terminal_cost(
        self,
        states: Float32Array,
        goal: Float32Array,
    ) -> Float32Array:
        """Evaluate batched terminal state cost (K,)."""

    @abstractmethod
    def compute_articulated_world_poses(self, state: Float32Array) -> Float32Array:
        """Compute packed float32 keypoint/link poses for telemetry & 3D rendering."""

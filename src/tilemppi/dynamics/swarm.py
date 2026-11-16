"""4-Drone Coordinated 3D Swarm dynamics with formation tracking & inter-agent avoidance."""

from __future__ import annotations

import numpy as np

from tilemppi.config import SDFConfig
from tilemppi.dynamics.base import RobotModel
from tilemppi.sdf.primitives import compute_barrier_cost, evaluate_sdf_batch
from tilemppi.types import Float32Array, RobotKind

SWARM_FORMATION_OFFSETS = np.array(
    [
        [0.65, 0.0, 0.15],
        [0.0, 0.65, -0.10],
        [0.0, -0.65, -0.10],
        [-0.65, 0.0, 0.05],
    ],
    dtype=np.float32,
)


class DroneSwarm(RobotModel):
    """4-Drone Coordinated 3D Swarm (state_dim=32, control_dim=16).

    State vector (32 floats = 4 drones x 8 floats):
      For drone i in {0, 1, 2, 3}, slice [8*i : 8*(i+1)]:
        [px, py, pz, vx, vy, vz, roll, pitch]

    Control vector (16 floats = 4 drones x 4 floats):
      For drone i in {0, 1, 2, 3}, slice [4*i : 4*(i+1)]:
        [thrust_accel, roll_cmd, pitch_cmd, climb_trim]
    """

    def __init__(
        self,
        gravity: float = 9.81,
        attitude_tau: float = 0.12,
        linear_drag: float = 0.25,
        min_inter_drone_dist: float = 0.52,
        repulsion_gain: float = 12.0,
    ) -> None:
        super().__init__(kind=RobotKind.SWARM, state_dim=32, control_dim=16)
        self.gravity = np.float32(gravity)
        self.inv_tau = np.float32(1.0 / attitude_tau)
        self.linear_drag = np.float32(linear_drag)
        self.min_inter_drone_dist = np.float32(min_inter_drone_dist)
        self.repulsion_gain = np.float32(repulsion_gain)

        u_min_single = np.array([2.0, -0.65, -0.65, -3.0], dtype=np.float32)
        u_max_single = np.array([20.0, 0.65, 0.65, 3.0], dtype=np.float32)
        self._u_min = np.tile(u_min_single, 4)
        self._u_max = np.tile(u_max_single, 4)

    @property
    def control_min(self) -> Float32Array:
        return self._u_min

    @property
    def control_max(self) -> Float32Array:
        return self._u_max

    def nominal_control(self) -> Float32Array:
        single = np.array([float(self.gravity), 0.0, 0.0, 0.0], dtype=np.float32)
        return np.tile(single, 4)

    def default_initial_state(
        self,
        position: tuple[float, float, float] = (-3.5, 0.0, 1.5),
    ) -> Float32Array:
        center = np.asarray(position, dtype=np.float32)
        state = np.zeros(32, dtype=np.float32)
        for i in range(4):
            state[8 * i : 8 * i + 3] = center + SWARM_FORMATION_OFFSETS[i]
        return state

    def extract_position(self, states: Float32Array) -> Float32Array:
        """Return swarm centroid (..., 3) across the 4 drones."""
        s = np.asarray(states, dtype=np.float32)
        p0 = s[..., 0:3]
        p1 = s[..., 8:11]
        p2 = s[..., 16:19]
        p3 = s[..., 24:27]
        return np.asarray(np.float32(0.25) * (p0 + p1 + p2 + p3), dtype=np.float32)

    def post_step_project(self, states: Float32Array) -> Float32Array:
        out = np.array(states, dtype=np.float32, copy=True)
        for i in range(4):
            out[..., 8 * i + 2] = np.maximum(out[..., 8 * i + 2], np.float32(0.20))
            out[..., 8 * i + 6] = np.clip(out[..., 8 * i + 6], np.float32(-0.85), np.float32(0.85))
            out[..., 8 * i + 7] = np.clip(out[..., 8 * i + 7], np.float32(-0.85), np.float32(0.85))
        return out

    def derivatives(
        self,
        states: Float32Array,
        controls: Float32Array,
        external_wrench: Float32Array | None = None,
    ) -> Float32Array:
        x = np.asarray(states, dtype=np.float32)
        u = self.clip_controls(np.asarray(controls, dtype=np.float32))
        dx = np.zeros_like(x, dtype=np.float32)

        positions = [x[..., 8 * i : 8 * i + 3] for i in range(4)]

        # Pairwise inter-drone potential repulsion accelerations
        rep_acc = [np.zeros_like(positions[0], dtype=np.float32) for _ in range(4)]
        r_safe = self.min_inter_drone_dist
        for i in range(4):
            for j in range(i + 1, 4):
                diff = positions[i] - positions[j]
                dist = np.sqrt(np.sum(diff * diff, axis=-1, keepdims=True) + np.float32(1e-6))
                overlap = np.maximum(np.float32(0.0), r_safe - dist)
                force_vec = (self.repulsion_gain * overlap / dist) * diff
                rep_acc[i] = rep_acc[i] + force_vec
                rep_acc[j] = rep_acc[j] - force_vec

        for i in range(4):
            base_s = 8 * i
            base_u = 4 * i
            vx = x[..., base_s + 3]
            vy = x[..., base_s + 4]
            vz = x[..., base_s + 5]
            roll = x[..., base_s + 6]
            pitch = x[..., base_s + 7]

            thrust = u[..., base_u + 0]
            roll_cmd = u[..., base_u + 1]
            pitch_cmd = u[..., base_u + 2]
            climb_trim = u[..., base_u + 3]

            ax = thrust * np.sin(pitch) - self.linear_drag * vx + rep_acc[i][..., 0]
            ay = -thrust * np.sin(roll) * np.cos(pitch) - self.linear_drag * vy + rep_acc[i][..., 1]
            az = (
                thrust * np.cos(roll) * np.cos(pitch)
                - self.gravity
                + np.float32(0.4) * climb_trim
                - self.linear_drag * vz
                + rep_acc[i][..., 2]
            )

            if external_wrench is not None:
                wrench = np.asarray(external_wrench, dtype=np.float32)
                ax = ax + wrench[..., 0]
                ay = ay + wrench[..., 1]
                az = az + wrench[..., 2]

            droll = self.inv_tau * (roll_cmd - roll)
            dpitch = self.inv_tau * (pitch_cmd - pitch)

            dx[..., base_s + 0] = vx
            dx[..., base_s + 1] = vy
            dx[..., base_s + 2] = vz
            dx[..., base_s + 3] = ax
            dx[..., base_s + 4] = ay
            dx[..., base_s + 5] = az
            dx[..., base_s + 6] = droll
            dx[..., base_s + 7] = dpitch

        return dx

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
        s = np.asarray(states, dtype=np.float32)
        total = np.zeros(s.shape[:-1], dtype=np.float32)
        positions = [s[..., 8 * i : 8 * i + 3] for i in range(4)]

        for i in range(4):
            pos_i = positions[i]
            vel_i = s[..., 8 * i + 3 : 8 * i + 6]
            att_i = s[..., 8 * i + 6 : 8 * i + 8]
            offset = SWARM_FORMATION_OFFSETS[i].reshape(1, 3)

            wp_err = pos_i - (waypoint.reshape(1, 3) + offset)
            goal_err = pos_i - (goal.reshape(1, 3) + offset)

            total = total + np.float32(5.5) * np.sum(wp_err * wp_err, axis=-1)
            total = total + np.float32(1.5) * np.sum(goal_err * goal_err, axis=-1)
            total = total + np.float32(0.25) * np.sum(vel_i * vel_i, axis=-1)
            total = total + np.float32(3.5) * np.sum(att_i * att_i, axis=-1)

            sdf_i = evaluate_sdf_batch(pos_i, obstacles_table, t=t)
            total = total + np.float32(0.45) * compute_barrier_cost(
                sdf_i, sdf_cfg, cvar_uncertainty_scale=cvar_scale
            )

        # Pairwise inter-drone collision barrier
        r_min = self.min_inter_drone_dist
        for i in range(4):
            for j in range(i + 1, 4):
                diff = positions[i] - positions[j]
                dist = np.sqrt(np.sum(diff * diff, axis=-1) + np.float32(1e-6))
                violation = np.maximum(np.float32(0.0), r_min - dist)
                total = total + np.float32(650.0) * (violation * violation)

        u_nom = self.nominal_control().reshape(1, 16)
        du = controls - prev_controls
        ctrl_dev = controls - u_nom
        total = total + np.float32(0.06) * np.sum(ctrl_dev * ctrl_dev, axis=-1)
        total = total + np.float32(0.18) * np.sum(du * du, axis=-1)
        return np.asarray(total, dtype=np.float32)

    def terminal_cost(
        self,
        states: Float32Array,
        goal: Float32Array,
    ) -> Float32Array:
        s = np.asarray(states, dtype=np.float32)
        total = np.zeros(s.shape[:-1], dtype=np.float32)
        for i in range(4):
            pos_i = s[..., 8 * i : 8 * i + 3]
            vel_i = s[..., 8 * i + 3 : 8 * i + 6]
            target_i = goal.reshape(1, 3) + SWARM_FORMATION_OFFSETS[i].reshape(1, 3)
            err = pos_i - target_i
            total = total + np.float32(28.0) * np.sum(err * err, axis=-1)
            total = total + np.float32(4.0) * np.sum(vel_i * vel_i, axis=-1)
        return np.asarray(total, dtype=np.float32)

    def compute_articulated_world_poses(self, state: Float32Array) -> Float32Array:
        """Return 36 floats: [centroid(3), 4x drone_state_8 (32), formation_flag(1)]."""
        s = np.asarray(state, dtype=np.float32).reshape(32)
        centroid = self.extract_position(s)
        poses = np.zeros(36, dtype=np.float32)
        poses[0:3] = centroid
        poses[3:35] = s
        poses[35] = 4.0
        return poses

"""6-DoF SE(3) Agile Quadrotor dynamics model (state_dim=16, control_dim=4)."""

from __future__ import annotations

import numpy as np

from tilemppi.config import SDFConfig
from tilemppi.dynamics.base import RobotModel
from tilemppi.sdf.primitives import compute_barrier_cost, evaluate_sdf_batch
from tilemppi.types import Float32Array, RobotKind


class Quadrotor3D(RobotModel):
    """6-DoF SE(3) quadrotor with unit quaternion attitude and 4-motor thrust mixing.

    State vector (16 floats, power-of-two padded):
      [0:3]   position [px, py, pz] (world frame, m)
      [3:7]   quaternion [qw, qx, qy, qz] (body-to-world unit quaternion)
      [7:10]  linear velocity [vx, vy, vz] (world frame, m/s)
      [10:13] angular velocity [wx, wy, wz] (body frame, rad/s)
      [13:16] filtered thrust / auxiliary state [aux0, aux1, aux2]

    Control vector (4 floats):
      [0:4]   individual motor thrusts [f0, f1, f2, f3] in Newtons [0, f_max]
    """

    def __init__(
        self,
        mass: float = 1.0,
        gravity: float = 9.81,
        arm_length: float = 0.18,
        yaw_moment_coeff: float = 0.022,
        inertia_diag: tuple[float, float, float] = (0.0085, 0.0085, 0.015),
        max_motor_thrust: float = 6.5,
        linear_drag: float = 0.0,
        angular_damping: float = 0.08,
    ) -> None:
        super().__init__(kind=RobotKind.QUADROTOR, state_dim=16, control_dim=4)
        self.mass = np.float32(mass)
        self.gravity = np.float32(gravity)
        self.arm_eff = np.float32(arm_length / np.sqrt(2.0))
        self.yaw_coeff = np.float32(yaw_moment_coeff)
        self.ixx = np.float32(inertia_diag[0])
        self.iyy = np.float32(inertia_diag[1])
        self.izz = np.float32(inertia_diag[2])
        self.max_motor_thrust = np.float32(max_motor_thrust)
        self.linear_drag = np.float32(linear_drag)
        self.angular_damping = np.float32(angular_damping)
        self._u_min = np.zeros(4, dtype=np.float32)
        self._u_max = np.full(4, self.max_motor_thrust, dtype=np.float32)

    @property
    def control_min(self) -> Float32Array:
        return self._u_min

    @property
    def control_max(self) -> Float32Array:
        return self._u_max

    def hover_motor_thrust(self) -> float:
        """Single-motor thrust required for static hover equilibrium."""
        return float(self.mass * self.gravity / np.float32(4.0))

    def nominal_control(self) -> Float32Array:
        return np.full(4, self.hover_motor_thrust(), dtype=np.float32)

    def default_initial_state(
        self,
        position: tuple[float, float, float] = (-3.5, 0.0, 1.5),
    ) -> Float32Array:
        state = np.zeros(16, dtype=np.float32)
        state[0:3] = np.asarray(position, dtype=np.float32)
        state[3] = 1.0  # qw = 1.0
        return state

    def post_step_project(self, states: Float32Array) -> Float32Array:
        out = np.array(states, dtype=np.float32, copy=True)
        q = out[..., 3:7]
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        safe_norm = np.where(norm > 1e-8, norm, np.float32(1.0))
        q_unit = np.where(
            norm > 1e-8,
            q / safe_norm,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        out[..., 3:7] = q_unit
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

        qw = x[..., 3]
        qx = x[..., 4]
        qy = x[..., 5]
        qz = x[..., 6]

        vx = x[..., 7]
        vy = x[..., 8]
        vz = x[..., 9]

        wx = x[..., 10]
        wy = x[..., 11]
        wz = x[..., 12]

        f0 = u[..., 0]
        f1 = u[..., 1]
        f2 = u[..., 2]
        f3 = u[..., 3]

        total_thrust = f0 + f1 + f2 + f3
        tau_x = self.arm_eff * (f0 - f1 - f2 + f3)
        tau_y = self.arm_eff * (f0 + f1 - f2 - f3)
        tau_z = self.yaw_coeff * (f0 - f1 + f2 - f3)

        # Body +Z unit vector in world frame from quaternion R(q)[:, 2]
        bz_x = np.float32(2.0) * (qx * qz + qw * qy)
        bz_y = np.float32(2.0) * (qy * qz - qw * qx)
        bz_z = qw * qw - qx * qx - qy * qy + qz * qz

        inv_m = np.float32(1.0) / self.mass
        ax = total_thrust * bz_x * inv_m - self.linear_drag * vx
        ay = total_thrust * bz_y * inv_m - self.linear_drag * vy
        az = total_thrust * bz_z * inv_m - self.gravity - self.linear_drag * vz

        if external_wrench is not None:
            wrench = np.asarray(external_wrench, dtype=np.float32)
            ax = ax + wrench[..., 0] * inv_m
            ay = ay + wrench[..., 1] * inv_m
            az = az + wrench[..., 2] * inv_m
            tau_x = tau_x + wrench[..., 3]
            tau_y = tau_y + wrench[..., 4]
            tau_z = tau_z + wrench[..., 5]

        # Quaternion kinematics: dq/dt = 0.5 * q (x) [0, wx, wy, wz]
        half = np.float32(0.5)
        dqw = half * (-qx * wx - qy * wy - qz * wz)
        dqx = half * (qw * wx + qy * wz - qz * wy)
        dqy = half * (qw * wy - qx * wz + qz * wx)
        dqz = half * (qw * wz + qx * wy - qy * wx)

        # Euler rigid-body rotational dynamics: I * dw/dt = tau - w x (I w) - d_ang * w
        dwx = (tau_x - (self.izz - self.iyy) * wy * wz - self.angular_damping * wx) / self.ixx
        dwy = (tau_y - (self.ixx - self.izz) * wz * wx - self.angular_damping * wy) / self.iyy
        dwz = (tau_z - (self.iyy - self.ixx) * wx * wy - self.angular_damping * wz) / self.izz

        dx[..., 0] = vx
        dx[..., 1] = vy
        dx[..., 2] = vz
        dx[..., 3] = dqw
        dx[..., 4] = dqx
        dx[..., 5] = dqy
        dx[..., 6] = dqz
        dx[..., 7] = ax
        dx[..., 8] = ay
        dx[..., 9] = az
        dx[..., 10] = dwx
        dx[..., 11] = dwy
        dx[..., 12] = dwz
        dx[..., 13] = (total_thrust - np.float32(4.0 * self.hover_motor_thrust())) * np.float32(0.5)
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
        pos = states[..., 0:3]
        quat = states[..., 3:7]
        vel = states[..., 7:10]
        omega = states[..., 10:13]

        wp_err = pos - waypoint.reshape(1, 3)
        goal_err = pos - goal.reshape(1, 3)
        wp_cost = np.float32(18.0) * np.sum(wp_err * wp_err, axis=-1)
        goal_pull = np.float32(4.0) * np.sum(goal_err * goal_err, axis=-1)

        # Upright tilt penalty: 1 - bz_z = 2*(qx^2 + qy^2)
        tilt_cost = np.float32(14.0) * (quat[..., 1] ** 2 + quat[..., 2] ** 2)
        vel_cost = np.float32(0.8) * np.sum(vel * vel, axis=-1)
        omega_cost = np.float32(0.45) * np.sum(omega * omega, axis=-1)

        u_nom = self.nominal_control().reshape(1, 4)
        ctrl_dev = controls - u_nom
        du = controls - prev_controls
        ctrl_cost = np.float32(0.15) * np.sum(ctrl_dev * ctrl_dev, axis=-1)
        smooth_cost = np.float32(0.35) * np.sum(du * du, axis=-1)

        # Ground floor protection + 3D SDF obstacle barrier
        ground_pen = np.maximum(np.float32(0.0), np.float32(0.35) - pos[..., 2])
        ground_cost = np.float32(450.0) * (ground_pen * ground_pen)

        sdf_d = evaluate_sdf_batch(pos, obstacles_table, t=t)
        coll_cost = compute_barrier_cost(sdf_d, sdf_cfg, cvar_uncertainty_scale=cvar_scale)

        return np.asarray(
            wp_cost
            + goal_pull
            + tilt_cost
            + vel_cost
            + omega_cost
            + ctrl_cost
            + smooth_cost
            + ground_cost
            + coll_cost,
            dtype=np.float32,
        )

    def terminal_cost(
        self,
        states: Float32Array,
        goal: Float32Array,
    ) -> Float32Array:
        pos = states[..., 0:3]
        quat = states[..., 3:7]
        vel = states[..., 7:10]
        omega = states[..., 10:13]
        err = pos - goal.reshape(1, 3)
        return np.asarray(
            np.float32(95.0) * np.sum(err * err, axis=-1)
            + np.float32(12.0) * np.sum(vel * vel, axis=-1)
            + np.float32(25.0) * (quat[..., 1] ** 2 + quat[..., 2] ** 2)
            + np.float32(2.5) * np.sum(omega * omega, axis=-1),
            dtype=np.float32,
        )

    def compute_articulated_world_poses(self, state: Float32Array) -> Float32Array:
        """Return 24 floats: [px,py,pz, qw,qx,qy,qz, vx,vy,vz, 4x rotor world xyz (12), thrust_norm (2)]."""
        s = np.asarray(state, dtype=np.float32).reshape(16)
        pos = s[0:3]
        qw, qx, qy, qz = float(s[3]), float(s[4]), float(s[5]), float(s[6])

        # Rotation matrix columns from unit quaternion
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qw * qz)
        r02 = 2.0 * (qx * qz + qw * qy)
        r10 = 2.0 * (qx * qy + qw * qz)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qw * qx)
        r20 = 2.0 * (qx * qz - qw * qy)
        r21 = 2.0 * (qy * qz + qw * qx)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        rot = np.array(
            [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
            dtype=np.float32,
        )

        arm = float(self.arm_eff)
        local_rotors = np.array(
            [
                [arm, arm, 0.0],
                [arm, -arm, 0.0],
                [-arm, -arm, 0.0],
                [-arm, arm, 0.0],
            ],
            dtype=np.float32,
        )
        world_rotors = pos.reshape(1, 3) + local_rotors @ rot.T

        poses = np.zeros(24, dtype=np.float32)
        poses[0:3] = pos
        poses[3:7] = s[3:7]
        poses[7:10] = s[7:10]
        poses[10:22] = world_rotors.reshape(12)
        poses[22] = s[13]
        poses[23] = 1.0
        return poses

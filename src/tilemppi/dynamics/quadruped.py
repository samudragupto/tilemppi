"""12-DoF Articulated Quadruped dynamics with batched link kinematics & ground contact."""

from __future__ import annotations

import numpy as np

from tilemppi.config import SDFConfig
from tilemppi.dynamics.base import RobotModel
from tilemppi.sdf.primitives import compute_barrier_cost, evaluate_sdf_batch
from tilemppi.types import Float32Array, RobotKind

# Hip offsets in torso body frame [x, y, z] for FL, FR, RL, RR
HIP_OFFSETS_BODY = np.array(
    [
        [0.24, 0.14, 0.0],  # Front-Left (FL)
        [0.24, -0.14, 0.0],  # Front-Right (FR)
        [-0.24, 0.14, 0.0],  # Rear-Left (RL)
        [-0.24, -0.14, 0.0],  # Rear-Right (RR)
    ],
    dtype=np.float32,
)

NOMINAL_HIP_FLEX = 0.65
NOMINAL_KNEE_FLEX = -1.30
LINK_THIGH = 0.264004  # 2 * 0.264004 * cos(0.65) == 0.4200 m standing height
LINK_CALF = 0.264004
NOMINAL_STANDING_Z = float(
    LINK_THIGH * np.cos(NOMINAL_HIP_FLEX) + LINK_CALF * np.cos(NOMINAL_HIP_FLEX + NOMINAL_KNEE_FLEX)
)


def compute_quadruped_link_kinematics(
    states: Float32Array,
) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array]:
    """Compute batched 3D world positions of (hips, knees, feet) and leg Jacobians.

    Args:
        states: Array of shape (..., 32).

    Returns:
        hips_world: (..., 4, 3)
        knees_world: (..., 4, 3)
        feet_world: (..., 4, 3)
        jacobians_body: (..., 4, 3, 3) mapping joint velocities to foot velocity in body frame
    """
    s = np.asarray(states, dtype=np.float32)
    orig_batch = s.shape[:-1]
    s_flat = s.reshape(-1, 32)
    n_batch = s_flat.shape[0]

    pos = s_flat[:, 0:3]
    qw = s_flat[:, 3]
    qx = s_flat[:, 4]
    qy = s_flat[:, 5]
    qz = s_flat[:, 6]

    # Rotation matrices R(q) of shape (N, 3, 3)
    r00 = np.float32(1.0) - np.float32(2.0) * (qy * qy + qz * qz)
    r01 = np.float32(2.0) * (qx * qy - qw * qz)
    r02 = np.float32(2.0) * (qx * qz + qw * qy)
    r10 = np.float32(2.0) * (qx * qy + qw * qz)
    r11 = np.float32(1.0) - np.float32(2.0) * (qx * qx + qz * qz)
    r12 = np.float32(2.0) * (qy * qz - qw * qx)
    r20 = np.float32(2.0) * (qx * qz - qw * qy)
    r21 = np.float32(2.0) * (qy * qz + qw * qx)
    r22 = np.float32(1.0) - np.float32(2.0) * (qx * qx + qy * qy)

    rot = np.stack(
        [
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    )  # (N, 3, 3)

    joints = s_flat[:, 14:26].reshape(n_batch, 4, 3)
    q_abd = joints[:, :, 0]
    q_hip = joints[:, :, 1]
    q_knee = joints[:, :, 2]
    q_sum = q_hip + q_knee

    c_abd = np.cos(q_abd)
    s_abd = np.sin(q_abd)
    c_hip = np.cos(q_hip)
    s_hip = np.sin(q_hip)
    c_sum = np.cos(q_sum)
    s_sum = np.sin(q_sum)

    l1 = np.float32(LINK_THIGH)
    l2 = np.float32(LINK_CALF)

    # Knee offset from hip in body frame
    knee_rel_x = -l1 * s_hip
    knee_rel_y = l1 * c_hip * s_abd
    knee_rel_z = -l1 * c_hip * c_abd
    knee_rel_body = np.stack([knee_rel_x, knee_rel_y, knee_rel_z], axis=-1)

    # Foot offset from hip in body frame
    leg_plane_z = l1 * c_hip + l2 * c_sum
    foot_rel_x = -(l1 * s_hip + l2 * s_sum)
    foot_rel_y = leg_plane_z * s_abd
    foot_rel_z = -leg_plane_z * c_abd
    foot_rel_body = np.stack([foot_rel_x, foot_rel_y, foot_rel_z], axis=-1)

    hip_body = np.broadcast_to(HIP_OFFSETS_BODY.reshape(1, 4, 3), (n_batch, 4, 3))
    knee_body = hip_body + knee_rel_body
    foot_body = hip_body + foot_rel_body

    # Batched matrix multiply: world = pos[:, None, :] + body @ R^T
    hips_world = pos[:, None, :] + np.einsum("nlj,nij->nli", hip_body, rot)
    knees_world = pos[:, None, :] + np.einsum("nlj,nij->nli", knee_body, rot)
    feet_world = pos[:, None, :] + np.einsum("nlj,nij->nli", foot_body, rot)

    # 3x3 analytical leg Jacobian J = d(foot_rel_body)/d(q_abd, q_hip, q_knee)
    zeros = np.zeros_like(q_abd)
    j00 = zeros
    j01 = -(l1 * c_hip + l2 * c_sum)
    j02 = -l2 * c_sum

    j10 = leg_plane_z * c_abd
    j11 = -(l1 * s_hip + l2 * s_sum) * s_abd
    j12 = -l2 * s_sum * s_abd

    j20 = leg_plane_z * s_abd
    j21 = (l1 * s_hip + l2 * s_sum) * c_abd
    j22 = l2 * s_sum * c_abd

    jac = np.stack(
        [
            np.stack([j00, j01, j02], axis=-1),
            np.stack([j10, j11, j12], axis=-1),
            np.stack([j20, j21, j22], axis=-1),
        ],
        axis=-2,
    )

    return (
        hips_world.reshape((*orig_batch, 4, 3)).astype(np.float32),
        knees_world.reshape((*orig_batch, 4, 3)).astype(np.float32),
        feet_world.reshape((*orig_batch, 4, 3)).astype(np.float32),
        jac.reshape((*orig_batch, 4, 3, 3)).astype(np.float32),
    )


class Quadruped12DoF(RobotModel):
    """12-DoF Articulated Quadruped with 3-link leg kinematics and spring-damper ground contact.

    State vector (32 floats, power-of-two padded):
      [0:3]   torso position [px, py, pz]
      [3:7]   torso quaternion [qw, qx, qy, qz]
      [7:10]  torso linear velocity [vx, vy, vz]
      [10:13] torso angular velocity [wx, wy, wz]
      [13]    gait phase phi
      [14:26] 12 joint angles (4 legs x [abd, hip, knee])
      [26:30] 4 foot ground contact indicators [0..1]
      [30:32] reserved padding

    Control vector (16 floats, power-of-two padded):
      [0:12]  joint target PD offsets (rad)
      [12:16] locomotion commands [vx_cmd, vy_cmd, wz_cmd, stance_boost]
    """

    def __init__(
        self,
        mass: float = 12.0,
        gravity: float = 9.81,
        kp_stance_z: float = 380.0,
        kd_stance_z: float = 48.0,
        kp_attitude: float = 65.0,
        kd_attitude: float = 14.0,
    ) -> None:
        super().__init__(kind=RobotKind.QUADRUPED, state_dim=32, control_dim=16)
        self.mass = np.float32(mass)
        self.gravity = np.float32(gravity)
        self.standing_z = np.float32(NOMINAL_STANDING_Z)
        self.kp_stance_z = np.float32(kp_stance_z)
        self.kd_stance_z = np.float32(kd_stance_z)
        self.kp_attitude = np.float32(kp_attitude)
        self.kd_attitude = np.float32(kd_attitude)
        self.ixx = np.float32(0.14)
        self.iyy = np.float32(0.42)
        self.izz = np.float32(0.48)

        u_min = np.full(16, -1.2, dtype=np.float32)
        u_max = np.full(16, 1.2, dtype=np.float32)
        u_min[12:16] = np.array([-3.0, -2.5, -3.0, -1.0], dtype=np.float32)
        u_max[12:16] = np.array([3.5, 2.5, 3.0, 1.5], dtype=np.float32)
        self._u_min = u_min
        self._u_max = u_max

        self._nominal_joints = np.tile(
            np.array([0.0, NOMINAL_HIP_FLEX, NOMINAL_KNEE_FLEX], dtype=np.float32),
            4,
        )

    @property
    def control_min(self) -> Float32Array:
        return self._u_min

    @property
    def control_max(self) -> Float32Array:
        return self._u_max

    def nominal_control(self) -> Float32Array:
        return np.zeros(16, dtype=np.float32)

    def default_initial_state(
        self,
        position: tuple[float, float, float] = (-3.5, 0.0, NOMINAL_STANDING_Z),
    ) -> Float32Array:
        state = np.zeros(32, dtype=np.float32)
        state[0] = np.float32(position[0])
        state[1] = np.float32(position[1])
        state[2] = np.float32(position[2] if abs(position[2] - 1.0) > 1e-4 else NOMINAL_STANDING_Z)
        state[3] = 1.0  # qw
        state[14:26] = self._nominal_joints
        state[26:30] = 1.0  # all 4 feet in contact at rest
        return state

    def post_step_project(self, states: Float32Array) -> Float32Array:
        out = np.array(states, dtype=np.float32, copy=True)
        q = out[..., 3:7]
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        safe_norm = np.where(norm > 1e-8, norm, np.float32(1.0))
        out[..., 3:7] = np.where(
            norm > 1e-8,
            q / safe_norm,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        out[..., 2] = np.maximum(out[..., 2], np.float32(0.12))
        out[..., 14:26] = np.clip(out[..., 14:26], np.float32(-2.2), np.float32(2.2))
        out[..., 26:30] = np.clip(out[..., 26:30], np.float32(0.0), np.float32(1.0))
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

        pz = x[..., 2]
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
        phi = x[..., 13]

        vx_cmd = u[..., 12]
        vy_cmd = u[..., 13]
        wz_cmd = u[..., 14]
        stance_boost = u[..., 15]

        # Heading yaw angle from quaternion
        siny_cosp = np.float32(2.0) * (qw * qz + qx * qy)
        cosy_cosp = np.float32(1.0) - np.float32(2.0) * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        # Desired world velocity from body locomotion command + joint stride synergy
        joint_stride_x = np.float32(0.25) * (u[..., 1] + u[..., 4] + u[..., 7] + u[..., 10])
        joint_stride_y = np.float32(0.25) * (u[..., 0] + u[..., 3] + u[..., 6] + u[..., 9])
        v_body_x = vx_cmd + np.float32(0.8) * joint_stride_x
        v_body_y = vy_cmd + np.float32(0.5) * joint_stride_y

        v_des_x = cy * v_body_x - sy * v_body_y
        v_des_y = sy * v_body_x + cy * v_body_y

        # Ground contact factor (1 when torso near or below standing height)
        contact_gate = np.clip(
            (np.float32(0.58) - pz) / np.float32(0.16),
            np.float32(0.0),
            np.float32(1.0),
        )

        # Horizontal traction force clamped to Coulomb friction cone
        k_trac = np.float32(95.0) * contact_gate
        fx_grf = k_trac * (v_des_x - vx)
        fy_grf = k_trac * (v_des_y - vy)

        # Vertical ground reaction spring-damper balancing gravity at standing_z
        fz_grf = contact_gate * (
            self.mass * self.gravity
            + self.kp_stance_z * (self.standing_z - pz)
            - self.kd_stance_z * vz
            + np.float32(18.0) * stance_boost
        )
        fz_grf = np.maximum(np.float32(0.0), fz_grf)

        inv_m = np.float32(1.0) / self.mass
        ax = fx_grf * inv_m
        ay = fy_grf * inv_m
        az = fz_grf * inv_m - self.gravity

        # Self-righting torso posture torque + yaw rate tracking
        tau_x = -self.kp_attitude * qx - self.kd_attitude * wx
        tau_y = -self.kp_attitude * qy - self.kd_attitude * wy
        tau_z = np.float32(8.5) * contact_gate * (wz_cmd - wz) - np.float32(1.5) * wz

        if external_wrench is not None:
            wrench = np.asarray(external_wrench, dtype=np.float32)
            ax = ax + wrench[..., 0] * inv_m
            ay = ay + wrench[..., 1] * inv_m
            az = az + wrench[..., 2] * inv_m
            tau_x = tau_x + wrench[..., 3]
            tau_y = tau_y + wrench[..., 4]
            tau_z = tau_z + wrench[..., 5]

        half = np.float32(0.5)
        dqw = half * (-qx * wx - qy * wy - qz * wz)
        dqx = half * (qw * wx + qy * wz - qz * wy)
        dqy = half * (qw * wy - qx * wz + qz * wx)
        dqz = half * (qw * wz + qx * wy - qy * wx)

        dwx = tau_x / self.ixx
        dwy = tau_y / self.iyy
        dwz = tau_z / self.izz

        # Trotting gait rhythm modulated by commanded speed
        cmd_speed = np.sqrt(v_body_x * v_body_x + v_body_y * v_body_y + wz_cmd * wz_cmd)
        gait_Gate = np.clip(cmd_speed / np.float32(0.5), np.float32(0.0), np.float32(1.0))
        dphi = np.float32(8.0) * gait_Gate

        # Diagonal trot phases: FL/RR at phi, FR/RL at phi + pi
        s_trot = np.sin(phi) * gait_Gate
        nom_j = self._nominal_joints

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
        dx[..., 13] = dphi

        # 12 joint angles relax toward (nominal + trot swing + u[0:12])
        k_joint = np.float32(12.0)
        for leg in range(4):
            phase_sign = np.float32(1.0 if leg in (0, 3) else -1.0)
            base_idx = 14 + 3 * leg
            u_idx = 3 * leg
            q_abd_target = nom_j[u_idx] + np.float32(0.35) * u[..., u_idx]
            q_hip_target = (
                nom_j[u_idx + 1]
                + np.float32(0.28) * phase_sign * s_trot
                + np.float32(0.45) * u[..., u_idx + 1]
            )
            q_knee_target = (
                nom_j[u_idx + 2]
                - np.float32(0.35) * np.maximum(np.float32(0.0), phase_sign * s_trot)
                + np.float32(0.45) * u[..., u_idx + 2]
            )
            dx[..., base_idx] = k_joint * (q_abd_target - x[..., base_idx])
            dx[..., base_idx + 1] = k_joint * (q_hip_target - x[..., base_idx + 1])
            dx[..., base_idx + 2] = k_joint * (q_knee_target - x[..., base_idx + 2])

            contact_target = contact_gate * np.where(
                phase_sign * s_trot <= np.float32(0.15),
                np.float32(1.0),
                np.float32(0.0),
            )
            dx[..., 26 + leg] = np.float32(10.0) * (contact_target - x[..., 26 + leg])

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

        # Target standing height is self.standing_z even if goal[2] is higher
        wp_target = np.array(
            [waypoint[0], waypoint[1], float(self.standing_z)],
            dtype=np.float32,
        )
        goal_target = np.array(
            [goal[0], goal[1], float(self.standing_z)],
            dtype=np.float32,
        )

        wp_err = pos - wp_target.reshape(1, 3)
        goal_err = pos - goal_target.reshape(1, 3)

        wp_cost = np.float32(22.0) * np.sum(wp_err * wp_err, axis=-1)
        goal_pull = np.float32(5.0) * np.sum(goal_err * goal_err, axis=-1)

        # Heading alignment toward waypoint
        dx_wp = waypoint[0] - pos[..., 0]
        dy_wp = waypoint[1] - pos[..., 1]
        dist_xy = np.sqrt(dx_wp * dx_wp + dy_wp * dy_wp)
        desired_yaw = np.arctan2(dy_wp, dx_wp)
        qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        yaw = np.arctan2(
            np.float32(2.0) * (qw * qz + qx * qy),
            np.float32(1.0) - np.float32(2.0) * (qy * qy + qz * qz),
        )
        yaw_diff = np.arctan2(np.sin(yaw - desired_yaw), np.cos(yaw - desired_yaw))
        heading_cost = np.where(
            dist_xy > np.float32(0.35),
            np.float32(4.5) * (yaw_diff * yaw_diff),
            np.float32(0.0),
        )

        # Posture & height stability
        height_err = pos[..., 2] - self.standing_z
        posture_cost = (
            np.float32(60.0) * (height_err * height_err)
            + np.float32(45.0) * (qx * qx + qy * qy)
            + np.float32(0.8) * np.sum(vel * vel, axis=-1)
            + np.float32(0.6) * np.sum(omega * omega, axis=-1)
        )

        du = controls - prev_controls
        ctrl_cost = np.float32(0.12) * np.sum(controls * controls, axis=-1)
        smooth_cost = np.float32(0.30) * np.sum(du * du, axis=-1)

        sdf_d = evaluate_sdf_batch(pos, obstacles_table, t=t)
        coll_cost = compute_barrier_cost(sdf_d, sdf_cfg, cvar_uncertainty_scale=cvar_scale)

        return np.asarray(
            wp_cost + goal_pull + heading_cost + posture_cost + ctrl_cost + smooth_cost + coll_cost,
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
        goal_target = np.array(
            [goal[0], goal[1], float(self.standing_z)],
            dtype=np.float32,
        )
        err = pos - goal_target.reshape(1, 3)
        return np.asarray(
            np.float32(110.0) * np.sum(err * err, axis=-1)
            + np.float32(14.0) * np.sum(vel * vel, axis=-1)
            + np.float32(55.0) * (quat[..., 1] ** 2 + quat[..., 2] ** 2),
            dtype=np.float32,
        )

    def compute_articulated_world_poses(self, state: Float32Array) -> Float32Array:
        """Return 50 floats: [torso_pos(3), quat(4), vel(3), hips(12), knees(12), feet(12), contacts(4)]."""
        s = np.asarray(state, dtype=np.float32).reshape(1, 32)
        hips, knees, feet, _ = compute_quadruped_link_kinematics(s)
        poses = np.zeros(50, dtype=np.float32)
        poses[0:3] = s[0, 0:3]
        poses[3:7] = s[0, 3:7]
        poses[7:10] = s[0, 7:10]
        poses[10:22] = hips.reshape(12)
        poses[22:34] = knees.reshape(12)
        poses[34:46] = feet.reshape(12)
        poses[46:50] = s[0, 26:30]
        return poses

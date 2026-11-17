"""Compact binary WebSocket telemetry frame encoder/decoder ('TMPI' magic header)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from tilemppi.types import Float32Array, StepResult

# Little-endian 32-byte header:
#   magic (uint32)          = 0x544D5049 ("TMPI")
#   version (uint16)        = 1
#   robot_id (uint16)       = 0: quadrotor, 1: quadruped, 2: swarm
#   frame_seq (uint32)      = step_index
#   sim_time (float32)      = simulation time (s)
#   num_vis_rollouts (u16)  = N_vis
#   horizon (uint16)        = H
#   num_obstacles (uint16)  = M_active
#   pose_floats (uint16)    = length of articulated_poses float32 array
#   step_ms (float32)       = control loop latency (ms)
#   ess_ratio (float32)     = ESS / K
TMPI_MAGIC = 0x544D5049
TMPI_VERSION = 1
TMPI_HEADER_STRUCT = struct.Struct("<IHHIfHHHHff")
TMPI_HEADER_BYTES = TMPI_HEADER_STRUCT.size  # 32 bytes


@dataclass(slots=True)
class DecodedTMPIFrame:
    """Unpacked representation of a binary TMPI telemetry frame."""

    version: int
    robot_id: int
    frame_seq: int
    sim_time: float
    num_vis_rollouts: int
    horizon: int
    num_obstacles: int
    pose_floats: int
    step_ms: float
    ess_ratio: float
    articulated_poses: Float32Array
    goal: Float32Array
    active_waypoint: Float32Array
    optimal_trajectory_xyz: Float32Array
    sampled_trajectories_xyz: Float32Array
    sampled_weights: Float32Array
    obstacles: Float32Array
    impulse_vec: Float32Array
    telemetry_vec: Float32Array


def _pack_active_obstacles12(obstacles_table: Float32Array) -> Float32Array:
    """Pack active obstacles from (MAX_OBS, 16) into (M_act, 12) for network streaming.

    Columns (12 floats):
      [0]    kind
      [1:4]  px, py, pz
      [4:7]  vx, vy, vz
      [7:10] param0, param1, param2
      [10:12] qx, qz (yaw/roll orientation components)
    """
    active_rows = obstacles_table[obstacles_table[:, 0] >= 0.0]
    m_act = active_rows.shape[0]
    if m_act == 0:
        return np.zeros((0, 12), dtype=np.float32)
    out = np.zeros((m_act, 12), dtype=np.float32)
    out[:, 0] = active_rows[:, 0]
    out[:, 1:4] = active_rows[:, 1:4]
    out[:, 4:7] = active_rows[:, 4:7]
    out[:, 7:10] = active_rows[:, 10:13]
    out[:, 10] = active_rows[:, 13]  # qx
    out[:, 11] = active_rows[:, 15]  # qz
    return out


def encode_tmpi_frame(result: StepResult) -> bytes:
    """Encode a StepResult into a compact little-endian binary 'TMPI' frame."""
    tel = result.telemetry
    poses = np.ascontiguousarray(result.articulated_poses, dtype="<f4").ravel()
    goal_wp = np.ascontiguousarray(
        np.concatenate([result.goal.ravel()[:3], result.active_waypoint.ravel()[:3]]),
        dtype="<f4",
    )
    opt_xyz = np.ascontiguousarray(result.optimal_trajectory_xyz, dtype="<f4")
    vis_xyz = np.ascontiguousarray(result.sampled_trajectories_xyz, dtype="<f4")
    vis_w = np.ascontiguousarray(result.sampled_weights, dtype="<f4").ravel()
    obs12 = np.ascontiguousarray(_pack_active_obstacles12(result.obstacles_table), dtype="<f4")
    imp6 = np.ascontiguousarray(result.impulse_vec, dtype="<f4").ravel()[:6]

    telemetry_vec = np.array(
        [
            tel.loop_hz,
            tel.l2_residency_pct,
            tel.vram_mb,
            tel.min_cost,
            tel.mean_cost,
            tel.min_sdf_clearance,
            float(tel.rollout_count),
            float(tel.backend.numeric_id),
        ],
        dtype="<f4",
    )

    n_vis = int(vis_xyz.shape[0])
    horizon = int(opt_xyz.shape[0])
    num_obs = int(obs12.shape[0])
    pose_floats = int(poses.shape[0])

    header = TMPI_HEADER_STRUCT.pack(
        TMPI_MAGIC,
        TMPI_VERSION,
        tel.robot.numeric_id,
        int(tel.step_index),
        float(tel.sim_time),
        n_vis,
        horizon,
        num_obs,
        pose_floats,
        float(tel.step_ms),
        float(tel.ess_ratio),
    )

    payload_parts = [
        header,
        poses.tobytes(),
        goal_wp.tobytes(),
        opt_xyz.tobytes(),
        vis_xyz.tobytes(),
        vis_w.tobytes(),
        obs12.tobytes(),
        imp6.tobytes(),
        telemetry_vec.tobytes(),
    ]
    return b"".join(payload_parts)


def decode_tmpi_frame(buffer: bytes | bytearray | memoryview) -> DecodedTMPIFrame:
    """Decode and validate a binary 'TMPI' frame."""
    raw = memoryview(buffer)
    if len(raw) < TMPI_HEADER_BYTES:
        msg = f"Truncated TMPI buffer: {len(raw)} bytes < {TMPI_HEADER_BYTES} byte header"
        raise ValueError(msg)

    (
        magic,
        version,
        robot_id,
        frame_seq,
        sim_time,
        n_vis,
        horizon,
        num_obs,
        pose_floats,
        step_ms,
        ess_ratio,
    ) = TMPI_HEADER_STRUCT.unpack_from(raw, 0)

    if magic != TMPI_MAGIC:
        msg = f"Invalid TMPI magic header: 0x{magic:08X} != 0x{TMPI_MAGIC:08X}"
        raise ValueError(msg)
    if version != TMPI_VERSION:
        msg = f"Unsupported TMPI protocol version: {version} != {TMPI_VERSION}"
        raise ValueError(msg)

    expected_floats = (
        pose_floats + 6 + (horizon * 3) + (n_vis * horizon * 3) + n_vis + (num_obs * 12) + 6 + 8
    )
    expected_bytes = TMPI_HEADER_BYTES + expected_floats * 4
    if len(raw) != expected_bytes:
        msg = f"TMPI payload byte mismatch: got {len(raw)} bytes, expected {expected_bytes}"
        raise ValueError(msg)

    floats = np.frombuffer(raw, dtype="<f4", offset=TMPI_HEADER_BYTES).astype(np.float32, copy=True)
    cursor = 0

    poses = floats[cursor : cursor + pose_floats]
    cursor += pose_floats

    goal = floats[cursor : cursor + 3]
    waypoint = floats[cursor + 3 : cursor + 6]
    cursor += 6

    opt_len = horizon * 3
    opt_xyz = floats[cursor : cursor + opt_len].reshape(horizon, 3)
    cursor += opt_len

    vis_len = n_vis * horizon * 3
    vis_xyz = floats[cursor : cursor + vis_len].reshape(n_vis, horizon, 3)
    cursor += vis_len

    vis_w = floats[cursor : cursor + n_vis]
    cursor += n_vis

    obs_len = num_obs * 12
    obs12 = floats[cursor : cursor + obs_len].reshape(num_obs, 12)
    cursor += obs_len

    imp6 = floats[cursor : cursor + 6]
    cursor += 6

    tel8 = floats[cursor : cursor + 8]

    return DecodedTMPIFrame(
        version=int(version),
        robot_id=int(robot_id),
        frame_seq=int(frame_seq),
        sim_time=float(sim_time),
        num_vis_rollouts=int(n_vis),
        horizon=int(horizon),
        num_obstacles=int(num_obs),
        pose_floats=int(pose_floats),
        step_ms=float(step_ms),
        ess_ratio=float(ess_ratio),
        articulated_poses=poses,
        goal=goal,
        active_waypoint=waypoint,
        optimal_trajectory_xyz=opt_xyz,
        sampled_trajectories_xyz=vis_xyz,
        sampled_weights=vis_w,
        obstacles=obs12,
        impulse_vec=imp6,
        telemetry_vec=tel8,
    )

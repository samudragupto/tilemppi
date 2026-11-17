import {
  BackendKind,
  DecodedTMPIFrame,
  ObstacleRecord,
  RobotKind,
} from './types';

export const TMPI_MAGIC = 0x544d5049; // "TMPI"
export const TMPI_VERSION = 1;
export const TMPI_HEADER_BYTES = 32;

const ROBOT_ID_MAP: Record<number, RobotKind> = {
  0: 'quadrotor',
  1: 'quadruped',
  2: 'swarm',
};

const BACKEND_ID_MAP: Record<number, BackendKind> = {
  0: 'auto',
  1: 'cutile',
  2: 'cutile_sim',
  3: 'triton',
  4: 'torch_gpu',
  5: 'torch_cpu',
  6: 'numpy_cpu',
};

export function decodeTMPIFrame(buffer: ArrayBuffer): DecodedTMPIFrame {
  if (buffer.byteLength < TMPI_HEADER_BYTES) {
    throw new Error(
      `Truncated TMPI buffer: ${buffer.byteLength} bytes < ${TMPI_HEADER_BYTES}`
    );
  }

  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== TMPI_MAGIC) {
    throw new Error(
      `Invalid TMPI magic header: 0x${magic.toString(16)} !== 0x${TMPI_MAGIC.toString(16)}`
    );
  }

  const version = view.getUint16(4, true);
  if (version !== TMPI_VERSION) {
    throw new Error(`Unsupported TMPI protocol version: ${version}`);
  }

  const robotId = view.getUint16(6, true);
  const frameSeq = view.getUint32(8, true);
  const simTime = view.getFloat32(12, true);
  const numVisRollouts = view.getUint16(16, true);
  const horizon = view.getUint16(18, true);
  const numObstacles = view.getUint16(20, true);
  const poseFloats = view.getUint16(22, true);
  const stepMs = view.getFloat32(24, true);
  const essRatio = view.getFloat32(28, true);

  const expectedFloats =
    poseFloats +
    6 +
    horizon * 3 +
    numVisRollouts * horizon * 3 +
    numVisRollouts +
    numObstacles * 12 +
    6 +
    8;
  const expectedBytes = TMPI_HEADER_BYTES + expectedFloats * 4;

  if (buffer.byteLength !== expectedBytes) {
    throw new Error(
      `TMPI payload byte mismatch: got ${buffer.byteLength} bytes, expected ${expectedBytes}`
    );
  }

  const floats = new Float32Array(buffer, TMPI_HEADER_BYTES, expectedFloats);
  let cursor = 0;

  const articulatedPoses = floats.slice(cursor, cursor + poseFloats);
  cursor += poseFloats;

  const goal: [number, number, number] = [
    floats[cursor],
    floats[cursor + 1],
    floats[cursor + 2],
  ];
  const activeWaypoint: [number, number, number] = [
    floats[cursor + 3],
    floats[cursor + 4],
    floats[cursor + 5],
  ];
  cursor += 6;

  const optLen = horizon * 3;
  const optimalTrajectoryXYZ = floats.slice(cursor, cursor + optLen);
  cursor += optLen;

  const visLen = numVisRollouts * horizon * 3;
  const sampledTrajectoriesXYZ = floats.slice(cursor, cursor + visLen);
  cursor += visLen;

  const sampledWeights = floats.slice(cursor, cursor + numVisRollouts);
  cursor += numVisRollouts;

  const obstacles: ObstacleRecord[] = [];
  for (let i = 0; i < numObstacles; i++) {
    const base = cursor + i * 12;
    obstacles.push({
      kind: Math.round(floats[base]),
      position: [floats[base + 1], floats[base + 2], floats[base + 3]],
      velocity: [floats[base + 4], floats[base + 5], floats[base + 6]],
      params: [floats[base + 7], floats[base + 8], floats[base + 9]],
      qx: floats[base + 10],
      qz: floats[base + 11],
    });
  }
  cursor += numObstacles * 12;

  const impulseVec: [number, number, number, number, number, number] = [
    floats[cursor],
    floats[cursor + 1],
    floats[cursor + 2],
    floats[cursor + 3],
    floats[cursor + 4],
    floats[cursor + 5],
  ];
  cursor += 6;

  const loopHz = floats[cursor];
  const l2ResidencyPct = floats[cursor + 1];
  const vramMb = floats[cursor + 2];
  const minCost = floats[cursor + 3];
  const meanCost = floats[cursor + 4];
  const minSdfClearance = floats[cursor + 5];
  const rolloutCount = Math.round(floats[cursor + 6]);
  const backendId = Math.round(floats[cursor + 7]);

  return {
    version,
    robotId,
    robotKind: ROBOT_ID_MAP[robotId] ?? 'quadrotor',
    frameSeq,
    simTime,
    numVisRollouts,
    horizon,
    numObstacles,
    poseFloats,
    stepMs,
    essRatio,
    articulatedPoses,
    goal,
    activeWaypoint,
    optimalTrajectoryXYZ,
    sampledTrajectoriesXYZ,
    sampledWeights,
    obstacles,
    impulseVec,
    telemetry: {
      loopHz,
      l2ResidencyPct,
      vramMb,
      minCost,
      meanCost,
      minSdfClearance,
      rolloutCount,
      backendId,
      backendName: BACKEND_ID_MAP[backendId] ?? 'torch_gpu',
    },
  };
}


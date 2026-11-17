import {
  DecodedTMPIFrame,
  ObstacleRecord,
  PlaygroundControlsState,
} from '../protocol/types';

export function computeMillionStepsPerSec(
  numRollouts: number,
  horizon: number,
  stepMs: number
): number {
  const safeMs = Math.max(stepMs, 0.05);
  return (numRollouts * horizon) / (safeMs * 1000.0);
}

export function formatL2BudgetMb(
  numRollouts: number,
  horizon: number,
  precision: 'fp32' | 'fp16' | 'fp8'
): { fusedMb: number; unfusedMb: number; residencyPct: number } {
  const bytesPerElem = precision === 'fp8' ? 1 : 2;
  const effDu = precision === 'fp32' ? 8 : 16;
  const fusedBytes =
    horizon * 16 * 4 +
    32 * 16 * 4 +
    2 * numRollouts * 4 +
    128 * horizon * 3 * 4 +
    numRollouts * horizon * effDu * bytesPerElem;
  const unfusedBytes =
    8 * numRollouts * horizon * 32 * 4 +
    numRollouts * (horizon + 1) * 32 * 4 +
    4 * numRollouts * horizon * 32 * 4 +
    2 * numRollouts * horizon * 16 * 4;
  const fusedMb = Number((fusedBytes / (1024 * 1024)).toFixed(2));
  const unfusedMb = Number((unfusedBytes / (1024 * 1024)).toFixed(1));
  const residencyPct = fusedMb <= 32.0 ? 100.0 : Number(((32.0 / fusedMb) * 100.0).toFixed(1));
  return { fusedMb, unfusedMb, residencyPct };
}

export function generateOfflinePreviewFrame(
  stepIndex: number,
  controls: PlaygroundControlsState,
  customGoal: [number, number, number],
  activeImpulse: [number, number, number, number, number, number],
  extraObstacles: ObstacleRecord[]
): DecodedTMPIFrame {
  const simTime = stepIndex * 0.025;
  const horizon = controls.horizon;
  const nVis = 64;
  const zNom = controls.robot === 'quadruped' ? 0.42 : 1.45;

  const px = -2.8 * Math.cos(simTime * 0.55);
  const py = 1.15 * Math.sin(simTime * 1.1);
  const pz = zNom + (controls.robot === 'quadruped' ? 0.02 * Math.abs(Math.sin(simTime * 6.0)) : 0.2 * Math.sin(simTime * 1.4));

  const obstacles: ObstacleRecord[] = [
    { kind: 1, position: [-1.4, 0.0, zNom], velocity: [0, 0, 0], params: [0.42, 1.1, 0], qx: 0, qz: 0 },
    { kind: 0, position: [0.0, 0.85 * Math.sin(simTime * 0.8), zNom], velocity: [0, 0.4, 0], params: [0.48, 0, 0], qx: 0, qz: 0 },
    { kind: 3, position: [1.3, 0.0, zNom], velocity: [0, 0, 0], params: [1.1, 0.2, 0], qx: 0, qz: 0 },
    ...extraObstacles,
  ];

  const optXYZ = new Float32Array(horizon * 3);
  const visXYZ = new Float32Array(nVis * horizon * 3);
  const visW = new Float32Array(nVis);

  let weightSum = 0;
  for (let k = 0; k < nVis; k++) {
    const spread = ((k / Math.max(1, nVis - 1)) - 0.5) * 2.0;
    const w = Math.exp(-2.8 * spread * spread);
    visW[k] = w;
    weightSum += w;

    for (let h = 0; h < horizon; h++) {
      const frac = (h + 1) / horizon;
      const ox = px + (customGoal[0] - px) * frac * 0.7;
      const oy = py + (customGoal[1] - py) * frac * 0.7 + Math.sin(frac * Math.PI) * spread * 1.15;
      const oz = pz + (customGoal[2] - pz) * frac * 0.6 + Math.sin(frac * Math.PI) * (k % 5 - 2) * 0.08;
      const base = (k * horizon + h) * 3;
      visXYZ[base] = ox;
      visXYZ[base + 1] = oy;
      visXYZ[base + 2] = Math.max(0.18, oz);
      if (k === Math.floor(nVis / 2)) {
        optXYZ[h * 3] = ox;
        optXYZ[h * 3 + 1] = py + (customGoal[1] - py) * frac * 0.7;
        optXYZ[h * 3 + 2] = Math.max(0.18, pz + (customGoal[2] - pz) * frac * 0.6);
      }
    }
  }
  for (let k = 0; k < nVis; k++) {
    visW[k] /= Math.max(weightSum, 1e-6);
  }

  let poses: Float32Array;
  let robotId = 0;
  if (controls.robot === 'quadrotor') {
    robotId = 0;
    poses = new Float32Array(24);
    poses[0] = px;
    poses[1] = py;
    poses[2] = pz;
    poses[3] = 1.0;
    const arm = 0.13;
    const offsets = [
      [arm, arm, 0],
      [arm, -arm, 0],
      [-arm, -arm, 0],
      [-arm, arm, 0],
    ];
    for (let i = 0; i < 4; i++) {
      poses[10 + i * 3] = px + offsets[i][0];
      poses[10 + i * 3 + 1] = py + offsets[i][1];
      poses[10 + i * 3 + 2] = pz + offsets[i][2];
    }
  } else if (controls.robot === 'quadruped') {
    robotId = 1;
    poses = new Float32Array(50);
    poses[0] = px;
    poses[1] = py;
    poses[2] = pz;
    poses[3] = 1.0;
    const hipOff = [
      [0.24, 0.14],
      [0.24, -0.14],
      [-0.24, 0.14],
      [-0.24, -0.14],
    ];
    for (let i = 0; i < 4; i++) {
      const phase = simTime * 7.0 + (i === 0 || i === 3 ? 0 : Math.PI);
      const swing = 0.08 * Math.sin(phase);
      const lift = Math.max(0, 0.06 * Math.sin(phase));
      const hx = px + hipOff[i][0];
      const hy = py + hipOff[i][1];
      poses[10 + i * 3] = hx;
      poses[10 + i * 3 + 1] = hy;
      poses[10 + i * 3 + 2] = pz;

      poses[22 + i * 3] = hx - 0.12 + swing * 0.5;
      poses[22 + i * 3 + 1] = hy;
      poses[22 + i * 3 + 2] = pz - 0.21;

      poses[34 + i * 3] = hx + swing;
      poses[34 + i * 3 + 1] = hy;
      poses[34 + i * 3 + 2] = lift;

      poses[46 + i] = lift < 0.01 ? 1.0 : 0.0;
    }
  } else {
    robotId = 2;
    poses = new Float32Array(36);
    poses[0] = px;
    poses[1] = py;
    poses[2] = pz;
    const form = [
      [0.65, 0, 0.15],
      [0, 0.65, -0.1],
      [0, -0.65, -0.1],
      [-0.65, 0, 0.05],
    ];
    for (let i = 0; i < 4; i++) {
      const b = 3 + i * 8;
      poses[b] = px + form[i][0];
      poses[b + 1] = py + form[i][1];
      poses[b + 2] = pz + form[i][2];
    }
  }

  const l2 = formatL2BudgetMb(controls.numRollouts, horizon, controls.precision);

  return {
    version: 1,
    robotId,
    robotKind: controls.robot,
    frameSeq: stepIndex,
    simTime,
    numVisRollouts: nVis,
    horizon,
    numObstacles: obstacles.length,
    poseFloats: poses.length,
    stepMs: 6.4,
    essRatio: 0.092,
    articulatedPoses: poses,
    goal: customGoal,
    activeWaypoint: customGoal,
    optimalTrajectoryXYZ: optXYZ,
    sampledTrajectoriesXYZ: visXYZ,
    sampledWeights: visW,
    obstacles,
    impulseVec: activeImpulse,
    telemetry: {
      loopHz: 156.2,
      l2ResidencyPct: l2.residencyPct,
      vramMb: 18.4,
      minCost: 42.5,
      meanCost: 118.0,
      minSdfClearance: 0.45,
      rolloutCount: controls.numRollouts,
      backendId: 4,
      backendName: controls.backend === 'auto' ? 'torch_gpu' : controls.backend,
    },
  };
}

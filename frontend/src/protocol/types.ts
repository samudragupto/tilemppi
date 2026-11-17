export type RobotKind = 'quadrotor' | 'quadruped' | 'swarm';
export type ScenarioKind =
  | 'obstacle_slalom'
  | 'moving_gates'
  | 'asteroid_storm'
  | 'kick_recovery'
  | 'u_trap_maze';
export type BackendKind =
  | 'auto'
  | 'cutile'
  | 'cutile_sim'
  | 'triton'
  | 'torch_gpu'
  | 'torch_cpu'
  | 'numpy_cpu';
export type PrecisionMode = 'fp32' | 'fp16' | 'fp8';

export interface ObstacleRecord {
  kind: number;
  position: [number, number, number];
  velocity: [number, number, number];
  params: [number, number, number];
  qx: number;
  qz: number;
}

export interface DecodedTMPIFrame {
  version: number;
  robotId: number;
  robotKind: RobotKind;
  frameSeq: number;
  simTime: number;
  numVisRollouts: number;
  horizon: number;
  numObstacles: number;
  poseFloats: number;
  stepMs: number;
  essRatio: number;
  articulatedPoses: Float32Array;
  goal: [number, number, number];
  activeWaypoint: [number, number, number];
  optimalTrajectoryXYZ: Float32Array;
  sampledTrajectoriesXYZ: Float32Array;
  sampledWeights: Float32Array;
  obstacles: ObstacleRecord[];
  impulseVec: [number, number, number, number, number, number];
  telemetry: {
    loopHz: number;
    l2ResidencyPct: number;
    vramMb: number;
    minCost: number;
    meanCost: number;
    minSdfClearance: number;
    rolloutCount: number;
    backendId: number;
    backendName: BackendKind;
  };
}

export interface PlaygroundControlsState {
  robot: RobotKind;
  scenario: ScenarioKind;
  backend: BackendKind;
  precision: PrecisionMode;
  numRollouts: number;
  horizon: number;
  temperature: number;
  noiseSigma: number;
  plannerEnabled: boolean;
  paused: boolean;
  slowMo: boolean;
  interactionMode: 'goal' | 'throw' | 'shove';
}


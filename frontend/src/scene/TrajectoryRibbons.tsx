import * as THREE from 'three';
import { DecodedTMPIFrame } from '../protocol/types';

export class TrajectoryRibbonsManager {
  public readonly group: THREE.Group;
  private readonly rolloutLines: THREE.LineSegments;
  private readonly rolloutGeo: THREE.BufferGeometry;
  private readonly optimalLine: THREE.Line;
  private readonly optimalGeo: THREE.BufferGeometry;
  private readonly goalBeacon: THREE.Mesh;
  private readonly waypointBeacon: THREE.Mesh;

  private readonly maxRollouts = 512;
  private readonly maxHorizon = 128;
  private readonly posBuffer: Float32Array;
  private readonly colorBuffer: Float32Array;
  private readonly optPosBuffer: Float32Array;

  constructor() {
    this.group = new THREE.Group();

    const maxSegments = this.maxRollouts * (this.maxHorizon - 1);
    this.posBuffer = new Float32Array(maxSegments * 2 * 3);
    this.colorBuffer = new Float32Array(maxSegments * 2 * 3);

    this.rolloutGeo = new THREE.BufferGeometry();
    this.rolloutGeo.setAttribute(
      'position',
      new THREE.BufferAttribute(this.posBuffer, 3)
    );
    this.rolloutGeo.setAttribute(
      'color',
      new THREE.BufferAttribute(this.colorBuffer, 3)
    );

    const rolloutMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.72,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.rolloutLines = new THREE.LineSegments(this.rolloutGeo, rolloutMat);
    this.group.add(this.rolloutLines);

    this.optPosBuffer = new Float32Array(this.maxHorizon * 3);
    this.optimalGeo = new THREE.BufferGeometry();
    this.optimalGeo.setAttribute(
      'position',
      new THREE.BufferAttribute(this.optPosBuffer, 3)
    );
    const optMat = new THREE.LineBasicMaterial({
      color: 0xffd740,
      linewidth: 3,
    });
    this.optimalLine = new THREE.Line(this.optimalGeo, optMat);
    this.group.add(this.optimalLine);

    const goalGeo = new THREE.OctahedronGeometry(0.24, 1);
    const goalMat = new THREE.MeshStandardMaterial({
      color: 0x00ffaa,
      emissive: 0x00cc88,
      emissiveIntensity: 0.9,
      wireframe: false,
    });
    this.goalBeacon = new THREE.Mesh(goalGeo, goalMat);
    this.group.add(this.goalBeacon);

    const wpGeo = new THREE.SphereGeometry(0.14, 16, 16);
    const wpMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 0.85,
    });
    this.waypointBeacon = new THREE.Mesh(wpGeo, wpMat);
    this.group.add(this.waypointBeacon);
  }

  public update(frame: DecodedTMPIFrame): void {
    const nVis = Math.min(frame.numVisRollouts, this.maxRollouts);
    const H = Math.min(frame.horizon, this.maxHorizon);
    const vis = frame.sampledTrajectoriesXYZ;
    const weights = frame.sampledWeights;

    let maxW = 1e-6;
    for (let k = 0; k < nVis; k++) {
      if (weights[k] > maxW) maxW = weights[k];
    }

    let segIdx = 0;
    for (let k = 0; k < nVis; k++) {
      const normW = Math.min(1.0, Math.pow(weights[k] / maxW, 0.38));
      // Elite trajectories glow electric cyan/emerald; rejected glow crimson/magenta
      const r = (1.0 - normW) * 0.92 + normW * 0.05;
      const g = (1.0 - normW) * 0.14 + normW * 0.96;
      const b = (1.0 - normW) * 0.55 + normW * 1.0;

      for (let h = 0; h < H - 1; h++) {
        const idxA = (k * frame.horizon + h) * 3;
        const idxB = (k * frame.horizon + h + 1) * 3;
        const vOff = segIdx * 6;

        this.posBuffer[vOff] = vis[idxA];
        this.posBuffer[vOff + 1] = vis[idxA + 1];
        this.posBuffer[vOff + 2] = vis[idxA + 2];
        this.posBuffer[vOff + 3] = vis[idxB];
        this.posBuffer[vOff + 4] = vis[idxB + 1];
        this.posBuffer[vOff + 5] = vis[idxB + 2];

        const fade = 1.0 - (h / H) * 0.45;
        this.colorBuffer[vOff] = r * fade;
        this.colorBuffer[vOff + 1] = g * fade;
        this.colorBuffer[vOff + 2] = b * fade;
        this.colorBuffer[vOff + 3] = r * fade;
        this.colorBuffer[vOff + 4] = g * fade;
        this.colorBuffer[vOff + 5] = b * fade;

        segIdx++;
      }
    }

    this.rolloutGeo.setDrawRange(0, segIdx * 2);
    (this.rolloutGeo.attributes.position as THREE.BufferAttribute).needsUpdate =
      true;
    (this.rolloutGeo.attributes.color as THREE.BufferAttribute).needsUpdate =
      true;

    const opt = frame.optimalTrajectoryXYZ;
    for (let h = 0; h < H; h++) {
      this.optPosBuffer[h * 3] = opt[h * 3];
      this.optPosBuffer[h * 3 + 1] = opt[h * 3 + 1];
      this.optPosBuffer[h * 3 + 2] = opt[h * 3 + 2] + 0.02;
    }
    this.optimalGeo.setDrawRange(0, H);
    (this.optimalGeo.attributes.position as THREE.BufferAttribute).needsUpdate =
      true;

    this.goalBeacon.position.set(frame.goal[0], frame.goal[1], frame.goal[2]);
    this.goalBeacon.rotation.z += 0.03;
    this.waypointBeacon.position.set(
      frame.activeWaypoint[0],
      frame.activeWaypoint[1],
      frame.activeWaypoint[2]
    );
  }
}

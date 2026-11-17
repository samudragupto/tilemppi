import * as THREE from 'three';
import { DecodedTMPIFrame, ObstacleRecord } from '../protocol/types';

export class ObstacleMeshesManager {
  public readonly group: THREE.Group;
  private readonly meshPool: THREE.Mesh[] = [];
  private readonly impulseArrow: THREE.ArrowHelper;

  constructor() {
    this.group = new THREE.Group();
    this.impulseArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 1),
      1.2,
      0xff3366,
      0.28,
      0.16
    );
    this.impulseArrow.visible = false;
    this.group.add(this.impulseArrow);
  }

  private buildPrimitiveMesh(obs: ObstacleRecord): THREE.Mesh {
    const [p0, p1, p2] = obs.params;
    let geo: THREE.BufferGeometry;
    let colorHex = 0xf43f5e;
    let emissiveHex = 0xbe123c;

    if (obs.kind === 0) {
      // Sphere
      geo = new THREE.SphereGeometry(Math.max(0.1, p0), 24, 24);
      colorHex = 0xf43f5e;
      emissiveHex = 0x9f1239;
    } else if (obs.kind === 1) {
      // Capsule aligned along local Z
      geo = new THREE.CapsuleGeometry(
        Math.max(0.1, p0),
        Math.max(0.1, p1 * 2.0),
        12,
        20
      );
      geo.rotateX(Math.PI / 2);
      colorHex = 0xfb7185;
      emissiveHex = 0xbe123c;
    } else if (obs.kind === 2) {
      // OrientedBox
      geo = new THREE.BoxGeometry(
        Math.max(0.1, p0 * 2.0),
        Math.max(0.1, p1 * 2.0),
        Math.max(0.1, p2 * 2.0)
      );
      colorHex = 0x6366f1;
      emissiveHex = 0x4338ca;
    } else if (obs.kind === 3) {
      // TorusGate (ring in local YZ plane -> rotateY(pi/2))
      geo = new THREE.TorusGeometry(
        Math.max(0.2, p0),
        Math.max(0.05, p1),
        16,
        36
      );
      geo.rotateY(Math.PI / 2);
      colorHex = 0x22d3ee;
      emissiveHex = 0x0891b2;
    } else {
      geo = new THREE.SphereGeometry(0.25, 16, 16);
    }

    const mat = new THREE.MeshStandardMaterial({
      color: colorHex,
      emissive: emissiveHex,
      emissiveIntensity: 0.45,
      metalness: 0.4,
      roughness: 0.35,
      transparent: true,
      opacity: 0.88,
    });
    return new THREE.Mesh(geo, mat);
  }

  public update(frame: DecodedTMPIFrame): void {
    const obsList = frame.obstacles;

    while (this.meshPool.length > obsList.length) {
      const m = this.meshPool.pop();
      if (m) {
        m.geometry.dispose();
        (m.material as THREE.Material).dispose();
        this.group.remove(m);
      }
    }

    for (let i = 0; i < obsList.length; i++) {
      const obs = obsList[i];
      if (i >= this.meshPool.length) {
        const mesh = this.buildPrimitiveMesh(obs);
        this.meshPool.push(mesh);
        this.group.add(mesh);
      } else {
        const existing = this.meshPool[i];
        if (existing.userData.kind !== obs.kind) {
          this.group.remove(existing);
          existing.geometry.dispose();
          const replacement = this.buildPrimitiveMesh(obs);
          this.meshPool[i] = replacement;
          this.group.add(replacement);
        }
      }
      const mesh = this.meshPool[i];
      mesh.userData.kind = obs.kind;
      mesh.position.set(obs.position[0], obs.position[1], obs.position[2]);
      const qw = Math.sqrt(
        Math.max(0, 1.0 - obs.qx * obs.qx - obs.qz * obs.qz)
      );
      mesh.quaternion.set(obs.qx, 0, obs.qz, qw);
    }

    const [fx, fy, fz] = frame.impulseVec;
    const fMag = Math.hypot(fx, fy, fz);
    if (fMag > 0.1 && frame.articulatedPoses.length >= 3) {
      const rx = frame.articulatedPoses[0];
      const ry = frame.articulatedPoses[1];
      const rz = frame.articulatedPoses[2];
      this.impulseArrow.position.set(rx, ry, rz + 0.15);
      this.impulseArrow.setDirection(
        new THREE.Vector3(fx / fMag, fy / fMag, fz / fMag)
      );
      this.impulseArrow.setLength(
        Math.min(2.2, 0.5 + fMag * 0.08),
        0.32,
        0.18
      );
      this.impulseArrow.visible = true;
    } else {
      this.impulseArrow.visible = false;
    }
  }
}

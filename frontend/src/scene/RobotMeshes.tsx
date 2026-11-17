import * as THREE from 'three';
import { DecodedTMPIFrame } from '../protocol/types';

export class RobotMeshesManager {
  public readonly group: THREE.Group;
  private readonly quadrotorGroup: THREE.Group;
  private readonly rotors: THREE.Mesh[] = [];
  private readonly quadrupedGroup: THREE.Group;
  private readonly quadrupedTorso: THREE.Mesh;
  private readonly hipNodes: THREE.Mesh[] = [];
  private readonly kneeNodes: THREE.Mesh[] = [];
  private readonly footNodes: THREE.Mesh[] = [];
  private readonly legLines: THREE.LineSegments;
  private readonly legLinePos: Float32Array;
  private readonly swarmGroup: THREE.Group;
  private readonly swarmDrones: THREE.Group[] = [];
  private readonly swarmLinks: THREE.LineSegments;
  private readonly swarmLinkPos: Float32Array;

  constructor() {
    this.group = new THREE.Group();

    // 1. 6-DoF Quadrotor
    this.quadrotorGroup = new THREE.Group();
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      metalness: 0.75,
      roughness: 0.25,
    });
    const coreMesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.18, 0.14, 0.06),
      coreMat
    );
    this.quadrotorGroup.add(coreMesh);

    const armMat = new THREE.MeshStandardMaterial({
      color: 0x475569,
      metalness: 0.6,
    });
    const arm1 = new THREE.Mesh(
      new THREE.CylinderGeometry(0.015, 0.015, 0.38, 10),
      armMat
    );
    arm1.rotation.z = Math.PI / 4;
    const arm2 = arm1.clone();
    arm2.rotation.z = -Math.PI / 4;
    this.quadrotorGroup.add(arm1, arm2);

    const rotorGeo = new THREE.CylinderGeometry(0.085, 0.085, 0.01, 20);
    rotorGeo.rotateX(Math.PI / 2);
    const rotorOffsets = [
      [0.13, 0.13, 0.03],
      [0.13, -0.13, 0.03],
      [-0.13, -0.13, 0.03],
      [-0.13, 0.13, 0.03],
    ];
    for (let i = 0; i < 4; i++) {
      const rMat = new THREE.MeshStandardMaterial({
        color: i < 2 ? 0x00f5d4 : 0x38bdf8,
        emissive: i < 2 ? 0x00bb99 : 0x0284c7,
        emissiveIntensity: 0.75,
        transparent: true,
        opacity: 0.75,
      });
      const rMesh = new THREE.Mesh(rotorGeo, rMat);
      rMesh.position.set(
        rotorOffsets[i][0],
        rotorOffsets[i][1],
        rotorOffsets[i][2]
      );
      this.rotors.push(rMesh);
      this.quadrotorGroup.add(rMesh);
    }
    this.group.add(this.quadrotorGroup);

    // 2. 12-DoF Articulated Quadruped
    this.quadrupedGroup = new THREE.Group();
    const torsoMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      emissive: 0x0ea5e9,
      emissiveIntensity: 0.25,
      metalness: 0.7,
      roughness: 0.3,
    });
    this.quadrupedTorso = new THREE.Mesh(
      new THREE.BoxGeometry(0.48, 0.24, 0.13),
      torsoMat
    );
    this.quadrupedGroup.add(this.quadrupedTorso);

    const jointGeo = new THREE.SphereGeometry(0.035, 12, 12);
    const hipMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 0.6,
    });
    for (let i = 0; i < 4; i++) {
      const hm = new THREE.Mesh(jointGeo, hipMat);
      const km = new THREE.Mesh(jointGeo, hipMat);
      const fm = new THREE.Mesh(
        new THREE.SphereGeometry(0.042, 12, 12),
        new THREE.MeshStandardMaterial({
          color: 0x10b981,
          emissive: 0x059669,
          emissiveIntensity: 0.9,
        })
      );
      this.hipNodes.push(hm);
      this.kneeNodes.push(km);
      this.footNodes.push(fm);
      this.quadrupedGroup.add(hm, km, fm);
    }

    // 4 legs x 2 segments (hip->knee, knee->foot) = 8 segments = 16 vertices
    this.legLinePos = new Float32Array(16 * 3);
    const legLineGeo = new THREE.BufferGeometry();
    legLineGeo.setAttribute(
      'position',
      new THREE.BufferAttribute(this.legLinePos, 3)
    );
    this.legLines = new THREE.LineSegments(
      legLineGeo,
      new THREE.LineBasicMaterial({ color: 0x7dd3fc, linewidth: 2 })
    );
    this.quadrupedGroup.add(this.legLines);
    this.group.add(this.quadrupedGroup);

    // 3. 4-Drone Swarm
    this.swarmGroup = new THREE.Group();
    for (let i = 0; i < 4; i++) {
      const dGroup = new THREE.Group();
      const body = new THREE.Mesh(
        new THREE.OctahedronGeometry(0.12, 1),
        new THREE.MeshStandardMaterial({
          color: 0x22d3ee,
          emissive: 0x0891b2,
          emissiveIntensity: 0.8,
        })
      );
      dGroup.add(body);
      this.swarmDrones.push(dGroup);
      this.swarmGroup.add(dGroup);
    }
    this.swarmLinkPos = new Float32Array(6 * 2 * 3);
    const linkGeo = new THREE.BufferGeometry();
    linkGeo.setAttribute(
      'position',
      new THREE.BufferAttribute(this.swarmLinkPos, 3)
    );
    this.swarmLinks = new THREE.LineSegments(
      linkGeo,
      new THREE.LineBasicMaterial({
        color: 0x00f5d4,
        transparent: true,
        opacity: 0.45,
      })
    );
    this.swarmGroup.add(this.swarmLinks);
    this.group.add(this.swarmGroup);
  }

  public update(frame: DecodedTMPIFrame): void {
    const p = frame.articulatedPoses;
    this.quadrotorGroup.visible = frame.robotKind === 'quadrotor';
    this.quadrupedGroup.visible = frame.robotKind === 'quadruped';
    this.swarmGroup.visible = frame.robotKind === 'swarm';

    if (frame.robotKind === 'quadrotor' && p.length >= 10) {
      this.quadrotorGroup.position.set(p[0], p[1], p[2]);
      // Quaternion in state is [qw, qx, qy, qz]
      this.quadrotorGroup.quaternion.set(p[4], p[5], p[6], p[3]);
      for (const r of this.rotors) {
        r.rotation.z += 0.45;
      }
    } else if (frame.robotKind === 'quadruped' && p.length >= 50) {
      this.quadrupedTorso.position.set(p[0], p[1], p[2]);
      this.quadrupedTorso.quaternion.set(p[4], p[5], p[6], p[3]);

      for (let i = 0; i < 4; i++) {
        const hx = p[10 + i * 3];
        const hy = p[11 + i * 3];
        const hz = p[12 + i * 3];

        const kx = p[22 + i * 3];
        const ky = p[23 + i * 3];
        const kz = p[24 + i * 3];

        const fx = p[34 + i * 3];
        const fy = p[35 + i * 3];
        const fz = p[36 + i * 3];

        const contact = p[46 + i] > 0.5;

        this.hipNodes[i].position.set(hx, hy, hz);
        this.kneeNodes[i].position.set(kx, ky, kz);
        this.footNodes[i].position.set(fx, fy, fz);

        const footMat = this.footNodes[i].material as THREE.MeshStandardMaterial;
        footMat.color.setHex(contact ? 0x10b981 : 0xf59e0b);
        footMat.emissive.setHex(contact ? 0x059669 : 0xd97706);

        const base = i * 12;
        this.legLinePos[base] = hx;
        this.legLinePos[base + 1] = hy;
        this.legLinePos[base + 2] = hz;
        this.legLinePos[base + 3] = kx;
        this.legLinePos[base + 4] = ky;
        this.legLinePos[base + 5] = kz;

        this.legLinePos[base + 6] = kx;
        this.legLinePos[base + 7] = ky;
        this.legLinePos[base + 8] = kz;
        this.legLinePos[base + 9] = fx;
        this.legLinePos[base + 10] = fy;
        this.legLinePos[base + 11] = fz;
      }
      (
        this.legLines.geometry.attributes.position as THREE.BufferAttribute
      ).needsUpdate = true;
    } else if (frame.robotKind === 'swarm' && p.length >= 35) {
      const coords: [number, number, number][] = [];
      for (let i = 0; i < 4; i++) {
        const b = 3 + i * 8;
        const dx = p[b];
        const dy = p[b + 1];
        const dz = p[b + 2];
        coords.push([dx, dy, dz]);
        this.swarmDrones[i].position.set(dx, dy, dz);
      }
      let linkIdx = 0;
      for (let i = 0; i < 4; i++) {
        for (let j = i + 1; j < 4; j++) {
          const off = linkIdx * 6;
          this.swarmLinkPos[off] = coords[i][0];
          this.swarmLinkPos[off + 1] = coords[i][1];
          this.swarmLinkPos[off + 2] = coords[i][2];
          this.swarmLinkPos[off + 3] = coords[j][0];
          this.swarmLinkPos[off + 4] = coords[j][1];
          this.swarmLinkPos[off + 5] = coords[j][2];
          linkIdx++;
        }
      }
      (
        this.swarmLinks.geometry.attributes.position as THREE.BufferAttribute
      ).needsUpdate = true;
    }
  }
}

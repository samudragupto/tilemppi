import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { DecodedTMPIFrame } from '../protocol/types';
import { TrajectoryRibbonsManager } from './TrajectoryRibbons';
import { RobotMeshesManager } from './RobotMeshes';
import { ObstacleMeshesManager } from './ObstacleMeshes';

interface PlaygroundCanvasProps {
  latestFrame: DecodedTMPIFrame | null;
  interactionMode: 'goal' | 'throw' | 'shove';
  onCanvasPointAction: (point: [number, number, number]) => void;
}

export const PlaygroundCanvas: React.FC<PlaygroundCanvasProps> = ({
  latestFrame,
  interactionMode,
  onCanvasPointAction,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const frameRef = useRef<DecodedTMPIFrame | null>(latestFrame);
  const modeRef = useRef(interactionMode);
  const actionCallbackRef = useRef(onCanvasPointAction);

  frameRef.current = latestFrame;
  modeRef.current = interactionMode;
  actionCallbackRef.current = onCanvasPointAction;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050814);
    scene.fog = new THREE.FogExp2(0x050814, 0.045);

    const camera = new THREE.PerspectiveCamera(
      52,
      container.clientWidth / Math.max(1, container.clientHeight),
      0.1,
      100
    );
    camera.up.set(0, 0, 1);

    let camTheta = -Math.PI * 0.42;
    let camPhi = 0.58;
    let camRadius = 9.2;
    const syncCamera = () => {
      camera.position.set(
        camRadius * Math.cos(camPhi) * Math.cos(camTheta),
        camRadius * Math.cos(camPhi) * Math.sin(camTheta),
        camRadius * Math.sin(camPhi)
      );
      camera.lookAt(0, 0, 1.0);
    };
    syncCamera();

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0x94a3b8, 1.15);
    scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0x38bdf8, 2.2);
    keyLight.position.set(6, -8, 12);
    scene.add(keyLight);

    const rimLight = new THREE.PointLight(0x00f5d4, 3.0, 20);
    rimLight.position.set(-4, 4, 5);
    scene.add(rimLight);

    // Arena Grid on XY plane (rotateX(PI/2) because Three GridHelper is XZ by default)
    const grid = new THREE.GridHelper(14, 28, 0x0ea5e9, 0x1e293b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const ribbons = new TrajectoryRibbonsManager();
    const robots = new RobotMeshesManager();
    const obstacles = new ObstacleMeshesManager();
    scene.add(ribbons.group, robots.group, obstacles.group);

    let dragging = false;
    let moved = false;
    let prevX = 0;
    let prevY = 0;

    const onPointerDown = (e: PointerEvent) => {
      dragging = true;
      moved = false;
      prevX = e.clientX;
      prevY = e.clientY;
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!dragging) return;
      const dx = e.clientX - prevX;
      const dy = e.clientY - prevY;
      if (Math.hypot(dx, dy) > 4) moved = true;
      camTheta -= dx * 0.0065;
      camPhi = Math.max(0.15, Math.min(1.42, camPhi + dy * 0.0055));
      prevX = e.clientX;
      prevY = e.clientY;
      syncCamera();
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      if (!moved) {
        const rect = renderer.domElement.getBoundingClientRect();
        const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
        const planeZ =
          frameRef.current?.robotKind === 'quadruped' ? 0.42 : 1.45;
        const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -planeZ);
        const hit = new THREE.Vector3();
        if (raycaster.ray.intersectPlane(plane, hit)) {
          const cx = Math.max(-4.5, Math.min(4.5, Number(hit.x.toFixed(2))));
          const cy = Math.max(-4.5, Math.min(4.5, Number(hit.y.toFixed(2))));
          actionCallbackRef.current([cx, cy, planeZ]);
        }
      }
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      camRadius = Math.max(3.5, Math.min(18.0, camRadius + e.deltaY * 0.005));
      syncCamera();
    };

    const dom = renderer.domElement;
    dom.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    dom.addEventListener('wheel', onWheel, { passive: false });

    const onResize = () => {
      if (!container) return;
      camera.aspect =
        container.clientWidth / Math.max(1, container.clientHeight);
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', onResize);

    let animId = 0;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      if (frameRef.current) {
        ribbons.update(frameRef.current);
        robots.update(frameRef.current);
        obstacles.update(frameRef.current);
      }
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      dom.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      dom.removeEventListener('wheel', onWheel);
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      if (dom.parentNode === container) {
        container.removeChild(dom);
      }
    };
  }, []);

  return <div ref={containerRef} className="playground-canvas" />;
};

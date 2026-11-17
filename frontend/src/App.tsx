import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ControlPanel } from './components/ControlPanel';
import { TelemetryHud } from './components/TelemetryHud';
import { decodeTMPIFrame } from './protocol/decoder';
import {
  DecodedTMPIFrame,
  ObstacleRecord,
  PlaygroundControlsState,
} from './protocol/types';
import { PlaygroundCanvas } from './scene/PlaygroundCanvas';
import { generateOfflinePreviewFrame } from './sim/offlinePreview';

export const App: React.FC = () => {
  const [controls, setControls] = useState<PlaygroundControlsState>({
    robot: 'quadrotor',
    scenario: 'obstacle_slalom',
    backend: 'auto',
    precision: 'fp32',
    numRollouts: 16384,
    horizon: 32,
    temperature: 1.2,
    noiseSigma: 0.45,
    plannerEnabled: true,
    paused: false,
    slowMo: false,
    interactionMode: 'goal',
  });

  const [latestFrame, setLatestFrame] = useState<DecodedTMPIFrame | null>(null);
  const [connected, setConnected] = useState<boolean>(false);
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const previewStepRef = useRef<number>(0);
  const previewGoalRef = useRef<[number, number, number]>([3.2, 0.0, 1.5]);
  const previewImpulseRef = useRef<[number, number, number, number, number, number]>([
    0, 0, 0, 0, 0, 0,
  ]);
  const previewImpulseStepsRef = useRef<number>(0);
  const previewExtraObsRef = useRef<ObstacleRecord[]>([]);

  // Connect to binary TMPI WebSocket
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    let socket: WebSocket | null = null;
    try {
      socket = new WebSocket(wsUrl);
      socket.binaryType = 'arraybuffer';
      wsRef.current = socket;

      socket.onopen = () => {
        setConnected(true);
      };

      socket.onmessage = (event: MessageEvent) => {
        if (event.data instanceof ArrayBuffer) {
          try {
            const decoded = decodeTMPIFrame(event.data);
            setLatestFrame(decoded);
            setLatencyHistory((prev) => [...prev.slice(-39), decoded.stepMs]);
          } catch {
            // Ignore malformed frame
          }
        }
      };

      socket.onclose = () => {
        setConnected(false);
      };

      socket.onerror = () => {
        setConnected(false);
      };
    } catch {
      setConnected(false);
    }

    return () => {
      if (socket && socket.readyState <= WebSocket.OPEN) {
        socket.close();
      }
    };
  }, []);

  // Client-side fallback preview loop when backend WS is not connected
  useEffect(() => {
    if (connected) return;
    const timer = window.setInterval(() => {
      if (controls.paused) return;
      previewStepRef.current += 1;
      if (previewImpulseStepsRef.current > 0) {
        previewImpulseStepsRef.current -= 1;
      } else {
        previewImpulseRef.current = [0, 0, 0, 0, 0, 0];
      }
      const frame = generateOfflinePreviewFrame(
        previewStepRef.current,
        controls,
        previewGoalRef.current,
        previewImpulseRef.current,
        previewExtraObsRef.current
      );
      setLatestFrame(frame);
      setLatencyHistory((prev) => [...prev.slice(-39), frame.stepMs]);
    }, 33);
    return () => window.clearInterval(timer);
  }, [connected, controls]);

  const handleUpdateControls = useCallback(
    (partial: Partial<PlaygroundControlsState>) => {
      setControls((prev) => {
        const next = { ...prev, ...partial };
        if (partial.paused !== undefined && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({ type: 'pause', paused: partial.paused })
          );
        }
        if (
          partial.robot !== undefined ||
          partial.scenario !== undefined ||
          partial.backend !== undefined ||
          partial.precision !== undefined ||
          partial.numRollouts !== undefined ||
          partial.horizon !== undefined ||
          partial.temperature !== undefined ||
          partial.noiseSigma !== undefined ||
          partial.plannerEnabled !== undefined
        ) {
          void fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              robot: next.robot,
              scenario: next.scenario,
              backend: next.backend,
              precision: next.precision,
              num_rollouts: next.numRollouts,
              horizon: next.horizon,
              temperature: next.temperature,
              noise_sigma: next.noiseSigma,
              planner_enabled: next.plannerEnabled,
            }),
          }).catch(() => undefined);
        }
        return next;
      });
    },
    []
  );

  const handleTriggerImpulse = useCallback(() => {
    const force: [number, number, number] =
      controls.robot === 'quadruped' ? [14.0, 38.0, 5.0] : [3.0, 8.5, 1.5];
    previewImpulseRef.current = [force[0], force[1], force[2], 0.2, -0.2, 0.1];
    previewImpulseStepsRef.current = 8;

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'impulse',
          force,
          torque: [0.2, -0.2, 0.1],
          duration_steps: 5,
        })
      );
    } else {
      void fetch('/api/impulse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          force,
          torque: [0.2, -0.2, 0.1],
          duration_steps: 5,
        }),
      }).catch(() => undefined);
    }
  }, [controls.robot]);

  const handleThrowObstacle = useCallback(() => {
    const zHeight = controls.robot === 'quadruped' ? 0.45 : 1.45;
    const pos: [number, number, number] = [0.2, 2.2, zHeight];
    const vel: [number, number, number] = [-0.4, -1.8, 0.0];
    previewExtraObsRef.current = [
      ...previewExtraObsRef.current.slice(-3),
      { kind: 0, position: pos, velocity: vel, params: [0.42, 0, 0], qx: 0, qz: 0 },
    ];

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'throw',
          position: pos,
          velocity: vel,
          radius: 0.42,
        })
      );
    } else {
      void fetch('/api/obstacles/throw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ position: pos, velocity: vel, radius: 0.42 }),
      }).catch(() => undefined);
    }
  }, [controls.robot]);

  const handleSingleStep = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'step' }));
    } else {
      previewStepRef.current += 1;
      setLatestFrame(
        generateOfflinePreviewFrame(
          previewStepRef.current,
          controls,
          previewGoalRef.current,
          previewImpulseRef.current,
          previewExtraObsRef.current
        )
      );
    }
  }, [controls]);

  const handleReset = useCallback(() => {
    previewStepRef.current = 0;
    previewExtraObsRef.current = [];
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reset' }));
    } else {
      void fetch('/api/reset', { method: 'POST' }).catch(() => undefined);
    }
  }, []);

  const handleCanvasPointAction = useCallback(
    (point: [number, number, number]) => {
      if (controls.interactionMode === 'goal') {
        previewGoalRef.current = point;
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'goal', goal: point }));
        }
      } else if (controls.interactionMode === 'throw') {
        const vel: [number, number, number] = [-point[0] * 0.6, -point[1] * 0.6, 0.0];
        previewExtraObsRef.current = [
          ...previewExtraObsRef.current.slice(-3),
          {
            kind: 0,
            position: point,
            velocity: vel,
            params: [0.4, 0, 0],
            qx: 0,
            qz: 0,
          },
        ];
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({
              type: 'throw',
              position: point,
              velocity: vel,
              radius: 0.4,
            })
          );
        }
      } else {
        handleTriggerImpulse();
      }
    },
    [controls.interactionMode, handleTriggerImpulse]
  );

  return (
    <main className="app-layout">
      <PlaygroundCanvas
        latestFrame={latestFrame}
        interactionMode={controls.interactionMode}
        onCanvasPointAction={handleCanvasPointAction}
      />
      <ControlPanel
        controls={controls}
        onUpdateControls={handleUpdateControls}
        onTriggerImpulse={handleTriggerImpulse}
        onThrowObstacle={handleThrowObstacle}
        onSingleStep={handleSingleStep}
        onReset={handleReset}
      />
      <TelemetryHud
        frame={latestFrame}
        connected={connected}
        precision={controls.precision}
        latencyHistory={latencyHistory}
      />
    </main>
  );
};

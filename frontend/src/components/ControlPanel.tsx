import React from 'react';
import {
  BackendKind,
  PlaygroundControlsState,
  PrecisionMode,
  RobotKind,
  ScenarioKind,
} from '../protocol/types';

interface ControlPanelProps {
  controls: PlaygroundControlsState;
  onUpdateControls: (partial: Partial<PlaygroundControlsState>) => void;
  onTriggerImpulse: () => void;
  onThrowObstacle: () => void;
  onSingleStep: () => void;
  onReset: () => void;
}

const ROLLOUT_OPTIONS = [256, 1024, 4096, 8192, 16384];
const HORIZON_OPTIONS = [16, 32, 64];

export const ControlPanel: React.FC<ControlPanelProps> = ({
  controls,
  onUpdateControls,
  onTriggerImpulse,
  onThrowObstacle,
  onSingleStep,
  onReset,
}) => {
  return (
    <aside className="panel control-panel">
      <div className="panel-header">
        <span className="badge-cutile">NVIDIA cuTile</span>
        <h2>TileMPPI Control Deck</h2>
      </div>

      <div className="control-group">
        <label>Robotic Platform</label>
        <div className="segmented-row">
          {(
            [
              ['quadrotor', 'Quadrotor 6-DoF'],
              ['quadruped', 'Quadruped 12-DoF'],
              ['swarm', '4-Drone Swarm'],
            ] as [RobotKind, string][]
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={controls.robot === id ? 'seg-btn active' : 'seg-btn'}
              onClick={() => onUpdateControls({ robot: id })}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="control-group">
        <label htmlFor="scenario-select">3D Arena Scenario</label>
        <select
          id="scenario-select"
          value={controls.scenario}
          onChange={(e) =>
            onUpdateControls({ scenario: e.target.value as ScenarioKind })
          }
        >
          <option value="obstacle_slalom">3D Obstacle Slalom</option>
          <option value="moving_gates">Moving Torus Gates</option>
          <option value="asteroid_storm">Dynamic Asteroid Storm</option>
          <option value="kick_recovery">Impulse Shove &amp; Kick Recovery</option>
          <option value="u_trap_maze">Concave U-Trap Maze (A* + MPPI)</option>
        </select>
      </div>

      <div className="control-row-2col">
        <div className="control-group">
          <label htmlFor="backend-select">Compute Backend</label>
          <select
            id="backend-select"
            value={controls.backend}
            onChange={(e) =>
              onUpdateControls({ backend: e.target.value as BackendKind })
            }
          >
            <option value="auto">Auto (Best GPU)</option>
            <option value="cutile">cuTile (@ct.kernel)</option>
            <option value="cutile_sim">cuTile Emulator</option>
            <option value="torch_gpu">Torch CUDA (sm_89)</option>
            <option value="triton">Triton Tile JIT</option>
            <option value="numpy_cpu">NumPy Golden CPU</option>
          </select>
        </div>

        <div className="control-group">
          <label>Tile Precision</label>
          <div className="segmented-row">
            {(['fp32', 'fp16', 'fp8'] as PrecisionMode[]).map((prec) => (
              <button
                key={prec}
                type="button"
                className={
                  controls.precision === prec ? 'seg-btn active' : 'seg-btn'
                }
                onClick={() => onUpdateControls({ precision: prec })}
              >
                {prec.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="control-group">
        <label>
          Parallel Universes (K):{' '}
          <strong>{controls.numRollouts.toLocaleString()}</strong>
        </label>
        <div className="segmented-row">
          {ROLLOUT_OPTIONS.map((k) => (
            <button
              key={k}
              type="button"
              className={
                controls.numRollouts === k ? 'seg-btn active' : 'seg-btn'
              }
              onClick={() => onUpdateControls({ numRollouts: k })}
            >
              {k >= 1024 ? `${k / 1024}K` : k}
            </button>
          ))}
        </div>
      </div>

      <div className="control-group">
        <label>
          Prediction Horizon (H): <strong>{controls.horizon} steps</strong>
        </label>
        <div className="segmented-row">
          {HORIZON_OPTIONS.map((h) => (
            <button
              key={h}
              type="button"
              className={controls.horizon === h ? 'seg-btn active' : 'seg-btn'}
              onClick={() => onUpdateControls({ horizon: h })}
            >
              H={h}
            </button>
          ))}
        </div>
      </div>

      <div className="control-row-2col">
        <div className="control-group">
          <label>
            Temp (&lambda;): <strong>{controls.temperature.toFixed(2)}</strong>
          </label>
          <input
            type="range"
            min={0.1}
            max={4.0}
            step={0.05}
            value={controls.temperature}
            onChange={(e) =>
              onUpdateControls({ temperature: parseFloat(e.target.value) })
            }
          />
        </div>
        <div className="control-group">
          <label>
            Noise (&sigma;): <strong>{controls.noiseSigma.toFixed(2)}</strong>
          </label>
          <input
            type="range"
            min={0.1}
            max={1.5}
            step={0.05}
            value={controls.noiseSigma}
            onChange={(e) =>
              onUpdateControls({ noiseSigma: parseFloat(e.target.value) })
            }
          />
        </div>
      </div>

      <div className="control-group">
        <label>3D Scene Click Action</label>
        <div className="segmented-row">
          {(
            [
              ['goal', 'Move Goal'],
              ['throw', 'Throw Sphere'],
              ['shove', 'Mouse Shove'],
            ] as ['goal' | 'throw' | 'shove', string][]
          ).map(([mode, label]) => (
            <button
              key={mode}
              type="button"
              className={
                controls.interactionMode === mode
                  ? 'seg-btn active'
                  : 'seg-btn'
              }
              onClick={() => onUpdateControls({ interactionMode: mode })}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="action-buttons-grid">
        <button
          type="button"
          className="action-btn btn-danger"
          onClick={onTriggerImpulse}
        >
          ⚡ Shove / Impulse Kick
        </button>
        <button
          type="button"
          className="action-btn btn-accent"
          onClick={onThrowObstacle}
        >
          ☄️ Launch Obstacle
        </button>
      </div>

      <div className="sim-transport-row">
        <button
          type="button"
          className="transport-btn"
          onClick={() => onUpdateControls({ paused: !controls.paused })}
        >
          {controls.paused ? '▶ Resume' : '⏸ Pause'}
        </button>
        <button
          type="button"
          className="transport-btn"
          onClick={onSingleStep}
        >
          ⏭ Step
        </button>
        <button
          type="button"
          className="transport-btn"
          onClick={() =>
            onUpdateControls({ plannerEnabled: !controls.plannerEnabled })
          }
        >
          A*: {controls.plannerEnabled ? 'ON' : 'OFF'}
        </button>
        <button type="button" className="transport-btn" onClick={onReset}>
          ↺ Reset
        </button>
      </div>
    </aside>
  );
};

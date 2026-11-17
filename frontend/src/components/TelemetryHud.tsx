import React from 'react';
import { DecodedTMPIFrame, PrecisionMode } from '../protocol/types';
import {
  computeMillionStepsPerSec,
  formatL2BudgetMb,
} from '../sim/offlinePreview';

interface TelemetryHudProps {
  frame: DecodedTMPIFrame | null;
  connected: boolean;
  precision: PrecisionMode;
  latencyHistory: number[];
}

export const TelemetryHud: React.FC<TelemetryHudProps> = ({
  frame,
  connected,
  precision,
  latencyHistory,
}) => {
  const numRollouts = frame?.telemetry.rolloutCount ?? 16384;
  const horizon = frame?.horizon ?? 32;
  const stepMs = frame?.stepMs ?? 6.5;
  const loopHz = frame?.telemetry.loopHz ?? 153.8;
  const mStepsSec = computeMillionStepsPerSec(numRollouts, horizon, stepMs);
  const l2Budget = formatL2BudgetMb(numRollouts, horizon, precision);
  const essPct = ((frame?.essRatio ?? 0.085) * 100.0).toFixed(1);
  const clearance = (frame?.telemetry.minSdfClearance ?? 0.42).toFixed(2);
  const backendLabel = (frame?.telemetry.backendName ?? 'torch_gpu').toUpperCase();

  // Build SVG sparkline path from latencyHistory
  const sparkPts = latencyHistory.length > 1 ? latencyHistory : [6.2, 6.4, 6.1, 6.5, 6.3];
  const maxLat = Math.max(15.0, ...sparkPts);
  const polyPoints = sparkPts
    .map((val, idx) => {
      const x = (idx / Math.max(1, sparkPts.length - 1)) * 220;
      const y = 42 - Math.min(38, (val / maxLat) * 38);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <aside className="panel telemetry-hud">
      <div className="hud-status-bar">
        <span className={connected ? 'status-dot online' : 'status-dot preview'} />
        <span className="status-text">
          {connected ? 'LIVE TMPI WEBSOCKET' : 'INTERACTIVE LOCAL PREVIEW'}
        </span>
        <span className="backend-pill">{backendLabel}</span>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <span className="metric-label">Control Frequency</span>
          <span className="metric-value cyan">{loopHz.toFixed(1)} Hz</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Kernel Latency</span>
          <span className="metric-value">{stepMs.toFixed(2)} ms</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Throughput</span>
          <span className="metric-value emerald">
            {mStepsSec.toFixed(1)} M-steps/s
          </span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Effective Sample Size</span>
          <span className="metric-value">{essPct}% ESS</span>
        </div>
      </div>

      <div className="l2-gauge-box">
        <div className="l2-gauge-header">
          <span>RTX 4060 32 MB L2 Cache Residency</span>
          <strong className="emerald">{l2Budget.residencyPct.toFixed(0)}% LOCKED</strong>
        </div>
        <div className="l2-progress-track">
          <div
            className="l2-progress-fill"
            style={{
              width: `${Math.min(100, (l2Budget.fusedMb / 32.0) * 100.0)}%`,
            }}
          />
        </div>
        <div className="l2-sub-stats">
          <span>Fused TileMPPI: {l2Budget.fusedMb} MB / 32.0 MB</span>
          <span className="muted">Unfused PyTorch: {l2Budget.unfusedMb} MB</span>
        </div>
      </div>

      <div className="metrics-row-compact">
        <div>
          <span className="metric-label">Min SDF Clearance</span>
          <strong className={Number(clearance) > 0.05 ? 'emerald' : 'danger'}>
            {clearance} m
          </strong>
        </div>
        <div>
          <span className="metric-label">Rollout Min Cost</span>
          <strong>{(frame?.telemetry.minCost ?? 42.0).toFixed(1)}</strong>
        </div>
        <div>
          <span className="metric-label">VRAM Footprint</span>
          <strong>{(frame?.telemetry.vramMb ?? 18.4).toFixed(1)} MB</strong>
        </div>
      </div>

      <div className="sparkline-container">
        <div className="sparkline-header">
          <span>Step Latency Trace (ms)</span>
          <span>Frame #{frame?.frameSeq ?? 0}</span>
        </div>
        <svg viewBox="0 0 220 46" className="sparkline-svg">
          <polyline
            fill="none"
            stroke="#00f5d4"
            strokeWidth="2"
            points={polyPoints}
          />
        </svg>
      </div>
    </aside>
  );
};

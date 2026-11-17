import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { PlaygroundControlsState } from '../protocol/types';
import {
  computeMillionStepsPerSec,
  formatL2BudgetMb,
  generateOfflinePreviewFrame,
} from '../sim/offlinePreview';

describe('Telemetry & L2 Cache Budget Helpers', () => {
  it('computes million state-steps per second accurately', () => {
    const mSteps = computeMillionStepsPerSec(16384, 32, 10.0);
    // 16384 * 32 = 524,288 steps in 0.01s => 52.4288 M-steps/s
    assert.ok(Math.abs(mSteps - 52.4288) < 1e-3);
  });

  it('keeps K=16384 fused L2 working set under 32 MB across fp32, fp16, and fp8', () => {
    for (const prec of ['fp32', 'fp16', 'fp8'] as const) {
      const budget = formatL2BudgetMb(16384, 32, prec);
      assert.ok(budget.fusedMb < 32.0);
      assert.ok(budget.unfusedMb > 100.0);
      assert.equal(budget.residencyPct, 100.0);
    }
  });

  it('generates valid offline preview frames for quadrotor, quadruped, and swarm', () => {
    const baseControls: PlaygroundControlsState = {
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
    };

    for (const robot of ['quadrotor', 'quadruped', 'swarm'] as const) {
      const frame = generateOfflinePreviewFrame(
        10,
        { ...baseControls, robot },
        [3.0, 0.0, 1.5],
        [0, 0, 0, 0, 0, 0],
        []
      );
      assert.equal(frame.robotKind, robot);
      assert.equal(frame.horizon, 32);
      assert.equal(frame.sampledWeights.length, frame.numVisRollouts);
      assert.ok(frame.articulatedPoses.length >= 24);
    }
  });
});

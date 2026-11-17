import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  decodeTMPIFrame,
  TMPI_HEADER_BYTES,
  TMPI_MAGIC,
  TMPI_VERSION,
} from '../protocol/decoder';

function buildValidTestFrameBuffer(): ArrayBuffer {
  const poseFloats = 24;
  const horizon = 8;
  const nVis = 4;
  const numObs = 2;

  const totalFloats =
    poseFloats +
    6 +
    horizon * 3 +
    nVis * horizon * 3 +
    nVis +
    numObs * 12 +
    6 +
    8;
  const totalBytes = TMPI_HEADER_BYTES + totalFloats * 4;
  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);

  view.setUint32(0, TMPI_MAGIC, true);
  view.setUint16(4, TMPI_VERSION, true);
  view.setUint16(6, 1, true); // quadruped
  view.setUint32(8, 42, true); // frameSeq
  view.setFloat32(12, 1.25, true); // simTime
  view.setUint16(16, nVis, true);
  view.setUint16(18, horizon, true);
  view.setUint16(20, numObs, true);
  view.setUint16(22, poseFloats, true);
  view.setFloat32(24, 5.5, true); // stepMs
  view.setFloat32(28, 0.12, true); // essRatio

  const floats = new Float32Array(buffer, TMPI_HEADER_BYTES, totalFloats);
  floats[0] = -1.5; // pose x
  floats[1] = 0.5; // pose y
  floats[2] = 0.42; // pose z

  // Goal & Waypoint at offset 24
  floats[24] = 3.0;
  floats[25] = -0.5;
  floats[26] = 0.42;
  floats[27] = 1.8;
  floats[28] = -0.2;
  floats[29] = 0.42;

  // Telemetry at the end (last 8 floats)
  const telBase = totalFloats - 8;
  floats[telBase] = 181.8; // loopHz
  floats[telBase + 1] = 100.0; // l2ResidencyPct
  floats[telBase + 2] = 18.5; // vramMb
  floats[telBase + 3] = 22.4; // minCost
  floats[telBase + 4] = 84.1; // meanCost
  floats[telBase + 5] = 0.38; // minSdfClearance
  floats[telBase + 6] = 16384; // rolloutCount
  floats[telBase + 7] = 4; // torch_gpu

  return buffer;
}

describe('TMPI Binary Protocol Decoder', () => {
  it('decodes a valid binary TMPI frame with byte-for-byte accuracy', () => {
    const buf = buildValidTestFrameBuffer();
    const frame = decodeTMPIFrame(buf);

    assert.equal(frame.version, 1);
    assert.equal(frame.robotId, 1);
    assert.equal(frame.robotKind, 'quadruped');
    assert.equal(frame.frameSeq, 42);
    assert.ok(Math.abs(frame.simTime - 1.25) < 1e-5);
    assert.equal(frame.numVisRollouts, 4);
    assert.equal(frame.horizon, 8);
    assert.equal(frame.numObstacles, 2);
    assert.equal(frame.poseFloats, 24);
    assert.ok(Math.abs(frame.goal[0] - 3.0) < 1e-5);
    assert.ok(Math.abs(frame.activeWaypoint[0] - 1.8) < 1e-5);
    assert.equal(frame.telemetry.rolloutCount, 16384);
    assert.equal(frame.telemetry.backendName, 'torch_gpu');
    assert.equal(frame.telemetry.l2ResidencyPct, 100.0);
  });

  it('rejects truncated headers, invalid magic bytes, and payload size mismatches', () => {
    const shortBuf = new ArrayBuffer(12);
    assert.throws(() => decodeTMPIFrame(shortBuf), /Truncated TMPI buffer/);

    const badMagicBuf = buildValidTestFrameBuffer();
    new DataView(badMagicBuf).setUint32(0, 0xdeadbeef, true);
    assert.throws(() => decodeTMPIFrame(badMagicBuf), /Invalid TMPI magic/);

    const badVerBuf = buildValidTestFrameBuffer();
    new DataView(badVerBuf).setUint16(4, 99, true);
    assert.throws(
      () => decodeTMPIFrame(badVerBuf),
      /Unsupported TMPI protocol version/
    );

    const mismatchBuf = buildValidTestFrameBuffer().slice(0, 48);
    assert.throws(
      () => decodeTMPIFrame(mismatchBuf),
      /TMPI payload byte mismatch/
    );
  });
});

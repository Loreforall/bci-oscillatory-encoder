import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { analyzeCove, flattenCode } from '../lib/cove.ts';
import { affine, parseCalibration, predictErp } from '../lib/cove-calibration.ts';
import { imageError, refineCode } from '../lib/cove-reconstruction.ts';

const load = url => JSON.parse(readFileSync(new URL(url, import.meta.url), 'utf8'));
const reference = load('./advanced-reference.json');
const renderer = load('../lib/cove-decoder.json');
const frames = Array.from({length: 5}, (_, t) => Uint8Array.from({length: 64**2 * 3}, (_, i) => {
  const p = Math.floor(i / 3), x = p % 64, y = Math.floor(p / 64);
  return [(13*x + 7*y + 17*t) % 256, (3*x + 11*y + 7*t) % 256,
    (5*x + 2*y + 29*t) % 256][i % 3];
}));
const {rows} = analyzeCove(frames, 24, [2, 0], true);

function close(actual, expected, tolerance) {
  assert.equal(actual.length, expected.length);
  assert.ok(Math.max(...actual.map((v, i) => Math.abs(v-expected[i]))) <= tolerance);
}

test('Fourier, causal Haar, complex dynamics and bypass agree with Python on RGB inputs', () => {
  rows.forEach((row, i) => close(row.spectrum, reference.spectrum[i], 2e-8));
  const row = rows[4];
  close(row.wavelets[12], reference.haar, 2e-8);
  close(flattenCode(row.state), reference.code, 2e-8);
  close(row.conventional, reference.feature, 2e-8);
  close(row.detail, reference.detail, 2e-8);
  close(row.pixels, reference.pixels, 2e-7);
});

test('inverse refinement agrees with Python and reduces image error', () => {
  const z = rows.map(row => flattenCode(row.state));
  const refined = refineCode(z, rows.map(row => row.pixels), rows.map(row => row.weights), renderer);
  const decoded = affine(refined[4], renderer.weights, renderer.offset);
  close(decoded, reference.reconstruction, 3e-7);
  assert.ok(imageError(rows[4].pixels, decoded).rmse <
    imageError(rows[4].pixels, affine(z[4], renderer.weights, renderer.offset)).rmse);
});

test('versioned calibration computes known ERP and rejects malformed feature blocks', () => {
  const offset = Array(240).fill(0.2);
  const mean = Array(rows[0].conventional.length).fill(0);
  const scale = Array(rows[0].conventional.length).fill(1);
  const model = {schema: 'cove-erp-v2', subject: 'sub-01',
    channels: ['O1','Oz','O2','PO7','PO3','POz','PO4','PO8'], coupling: [2,0],
    mode: 'visual_kernel', width: 2, mixture: [1,0,0], fit_conditions: 1, test_conditions: 40,
    blocks: [{mean, scale, anchors: [mean]}, null, null], held_out: {rmse:.1, skill:.1, skill_ci95:[0,.2]},
    offset, dual: [Array(240).fill(0.4)]};
  const values = predictErp(parseCalibration(JSON.stringify(model)), scale, [], []);
  close(values, Array(240).fill(.2+.4*Math.exp(-.5)), 1e-12);
  model.blocks[0].scale[249] = 0;
  assert.throws(() => parseCalibration(JSON.stringify(model)), /valid COVE ERP calibration/);
});

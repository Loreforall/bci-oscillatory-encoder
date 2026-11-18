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

test('field exports preserve the full clock, complex samples, graph, and sparse phase', async () => {
  const {fieldJson, fieldCsv} = await import('../lib/cove-export.ts');
  const analysis = analyzeCove(frames, 24, [2, 0], true);
  const source = {kind:'images', fps:24, sampling:'filename-order', processed_frames_sha256:'a'.repeat(64)};
  const report = JSON.parse(await fieldJson(analysis, source, [2, 0], null).text());
  assert.equal(report.schema, 'cove-field-v1');
  assert.deepEqual(report.internal_complex_shape, [25, 25, 4, 2]);
  assert.equal(report.internal_first_time_s, 1/120);
  assert.equal(report.internal_complex_flat.length, 5000);
  const site12 = Array.from({length: 25}, (_, t) =>
    report.internal_complex_flat.slice(t*200+12*8, t*200+13*8)).flat();
  close(site12, reference.site_12_trace, 2e-8);
  report.frames.forEach((frame, t) => {
    assert.equal(frame.input_sample_time_s, t/24);
    assert.equal(frame.state_time_s, (t+1)/24);
    frame.spatial_weights.forEach(edges => close([edges.reduce((s,e)=>s+e.weight,0)], [1], 1e-12));
    assert.ok(frame.spatial_dirichlet_energy.every(energy => energy >= 0));
    const values = frame.states.flatMap(modes => modes.flatMap(({re,im})=>[re,im]));
    close(values, report.internal_complex_flat.slice((t*5+4)*200, (t+1)*5*200), 0);
  });
  const csv = (await fieldCsv(analysis, source).text()).trim().split('\n');
  assert.equal(csv.length, 501);
  assert.equal(csv[0].split(',').length, 19);
  assert.equal(csv[1].split(',').length, 19);
  const still = analyzeCove(frames.slice(0,1), 7, [2,0], true);
  const stillJson = JSON.parse(await fieldJson(still, {...source,fps:7}, [2,0], null).text());
  assert.equal(stillJson.model_frame_rate_hz, 24);
  assert.equal(stillJson.frames[0].state_time_s, 5/24);
});

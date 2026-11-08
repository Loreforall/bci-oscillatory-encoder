import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { analyze, encodeFeatures, synthesize } from '../lib/encoder.ts';
import { extractFeatures, motionSpectrum, SIZE } from '../lib/features.ts';

const fixture = JSON.parse(readFileSync(new URL('./python-reference.json', import.meta.url), 'utf8'));

function near(actual, expected, tolerance, path = 'value') {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, path);
    expected.forEach((value, i) => near(actual[i], value, tolerance, `${path}[${i}]`));
    return;
  }
  assert.ok(Math.abs(actual - expected) <= tolerance, `${path}: ${actual} vs ${expected}`);
}

const pixels = makePixelFixtures();
function makePixelFixtures() {
  const patterns = [
    () => [240, 33, 110],
    x => Array(3).fill(x >= 25 ? 220 : 30),
    (x, y) => Array(3).fill(x + y >= 63 ? 230 : 20),
    (x, y) => [(3*x + 5*y) % 256, (11*x + 2*y) % 256, (7*x + 13*y) % 256],
    (x, y) => Array(3).fill((Math.floor(x/3) + Math.floor(y/3)) % 2 ? 210 : 35),
  ];
  return patterns.map(rgb => Uint8Array.from({length: SIZE**2 * 3}, (_, i) => {
    const pixel = Math.floor(i / 3);
    return rgb(pixel % SIZE, Math.floor(pixel / SIZE))[i % 3];
  }));
}

test('visual features and temporal spectrum agree on identical processed RGB pixels', () => {
  const features = extractFeatures(pixels);
  near(features, fixture.cases[0].features, 2e-6, 'features');
  near(extractFeatures([pixels[0]])[0].slice(8), Array(8).fill(0), 0);
  const spectrum = motionSpectrum(features, 24);
  near(spectrum.hz, fixture.motion_hz, 1e-12);
  near(spectrum.power, fixture.motion_power, 2e-7);
});

for (const reference of fixture.cases) {
  test(`NumPy template and Torch synthesis parity at ${reference.channels} channels, ${reference.sample_rate} Hz`, () => {
    const template = encodeFeatures(reference.features, reference.channels);
    for (const key of Object.keys(template)) near(template[key], reference.template[key], 1e-6, key);
    const wave = synthesize(template, reference.sample_rate);
    assert.equal(wave[0][0].length, Math.floor(reference.sample_rate * 0.5));
    near(wave.map(frame => frame.map(channel => [channel[0], channel[1], channel[13], channel.at(-1)])),
      reference.waveform_samples, 5e-6);
    const samples = wave.flat(2);
    near(Math.sqrt(samples.reduce((sum, sample) => sum + sample**2, 0) / samples.length),
      reference.waveform_rms, 5e-6);
    near(samples.reduce((peak, sample) => Math.max(peak, Math.abs(sample)), 0), reference.waveform_peak, 5e-6);
  });
}

test('end-to-end analysis exports provenance and the computed template', async () => {
  const result = await analyze(pixels, {kind: 'images', width: SIZE, height: SIZE, fps: 24, sampling: 'filename-order'}, 3, 101);
  assert.match(result.input.processed_frames_sha256, /^[0-9a-f]{64}$/);
  assert.equal(result.waveforms.length, pixels.length);
  near(result.metrics.waveform_rms, fixture.cases[0].waveform_rms, 5e-6);
});

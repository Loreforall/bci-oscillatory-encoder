import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { analyzeCove, spectrogram } from '../lib/cove.ts';

const fixture = JSON.parse(readFileSync(new URL('./cove-reference.json', import.meta.url), 'utf8'));
const frames = [0, .7, 1.4].map(phase => Uint8Array.from({length: 64**2 * 3}, (_, i) =>
  Math.trunc(120 + 80 * Math.cos(.4*(Math.floor(i/3) % 64) + .32*Math.floor(i/192) + phase))));
const code = row => row.state.flatMap(modes => [...modes.map(pair => pair[0]), ...modes.map(pair => pair[1])]);
function close(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 2e-8, `${i}: ${value} vs ${expected[i]}`));
}

test('COVE matches Python distributed states and coupling on shared video pixels', () => {
  const {rows, internal, internalRate} = analyzeCove(frames, 13, [1, 3]);
  rows.forEach((row, i) => {
    close(code(row), fixture.video_code[i]);
    close(Array.from({length: 25}, (_, j) => row.weights[12].find(edge => edge.j === j)?.weight ?? 0),
      fixture.selected_weights[i]);
  });
  assert.ok(internalRate >= 100);
  assert.ok(spectrogram(internal.map(row => row[12]), internalRate).flat().every(Number.isFinite));
});

test('still image uses the Python held-drive calibration protocol', () => {
  const {rows} = analyzeCove(frames.slice(0, 1), 10, [1, 3]);
  close(code(rows[0]), fixture.still_code);
});

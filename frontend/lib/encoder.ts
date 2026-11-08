import { extractFeatures, FEATURE_NAMES, motionSpectrum, SIZE } from './features.ts';

export const BANDS = [6, 10, 20, 40];
export const WINDOW_SECONDS = 0.5;

export function encodeFeatures(features: number[][], channels: number) {
  if (!Number.isInteger(channels) || channels < 1 || channels > 16) {
    throw new Error('Choose an integer channel count from 1 to 16.');
  }
  if (!features.length || features.some(row => row.length !== 16 || !row.every(Number.isFinite))) {
    throw new Error('Expected a nonempty sequence of 16 finite features per frame.');
  }
  const band_power: number[][][] = [];
  const band_phase: number[][][] = [];
  const spatial_weights: number[][] = [];
  for (const row of features) {
    const [brightness, contrast, edges, , high, foveal, peripheral, motion] =
      row.slice(0, 8).map(value => Math.max(0, Math.min(1, value)));
    const drivers = [motion, brightness, contrast, (edges + high) / 2];
    const phase = [motion, brightness, contrast, edges].map(value => Math.PI * (2 * value - 1));
    const positions = Array.from({length: channels}, (_, channel) => channels === 1 ? 0 : channel / (channels - 1));
    const preference = positions.map(position => {
      const edge = Math.abs(2 * position - 1);
      return foveal * (1 - edge) + peripheral * edge + 0.05;
    });
    const total = preference.reduce((sum, value) => sum + value, 0);
    band_power.push(positions.map(position => drivers.map(value => (0.15 + 0.6 * value) * (0.8 + 0.4 * position))));
    band_phase.push(positions.map(() => [...phase]));
    spatial_weights.push(preference.map(value => value / total));
  }
  return { band_power, band_phase, spatial_weights };
}

export function synthesize(template: ReturnType<typeof encodeFeatures>, sampleRate: number) {
  if (!Number.isInteger(sampleRate) || sampleRate < 100 || sampleRate > 500) {
    throw new Error('Choose an integer synthesis rate from 100 to 500 Hz.');
  }
  const samples = Math.floor(sampleRate * WINDOW_SECONDS);
  return template.band_power.map((frame, f) => frame.map((powers, c) => {
    const phase = template.band_phase[f][c];
    const weight = template.spatial_weights[f][c];
    return Array.from({length: samples}, (_, sample) => weight * powers.reduce((sum, power, band) =>
      sum + power * Math.sin(2 * Math.PI * BANDS[band] * sample / sampleRate + phase[band]), 0));
  }));
}

export type Input = {
  kind: 'synthetic' | 'images' | 'video';
  width: number;
  height: number;
  fps: number;
  sampling: string;
  duration_seconds?: number;
};

export async function analyze(frames: Uint8Array[], input: Input, channels: number, sampleRate: number) {
  const features = extractFeatures(frames);
  const template = encodeFeatures(features, channels);
  const waveforms = synthesize(template, sampleRate);
  let sumSquares = 0;
  let peak = 0;
  for (const frame of waveforms) {
    for (const channel of frame) {
      for (const sample of channel) {
        sumSquares += sample ** 2;
        peak = Math.max(peak, Math.abs(sample));
      }
    }
  }
  const meanPower = BANDS.map((_, band) => template.band_power.reduce((sum, frame) =>
    sum + frame.reduce((channelSum, channel) => channelSum + channel[band], 0), 0) / (frames.length * channels));

  const packed = new Uint8Array(frames.length * SIZE * SIZE * 3);
  frames.forEach((frame, i) => packed.set(frame, i * SIZE * SIZE * 3));
  const digest = await crypto.subtle.digest('SHA-256', packed);
  const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  return {
    schema_version: 2,
    signal_type: 'simulated oscillations',
    input: {...input, frames: frames.length, processed_frames_sha256: hash},
    parameters: {encoder: 'baseline', channels, bands_hz: BANDS, sample_rate_hz: sampleRate,
      window_seconds: WINDOW_SECONDS, feature_size: [SIZE, SIZE]},
    runtime: {engine_revision: 'browser-baseline-v1', browser: navigator.userAgent},
    checkpoint: null,
    feature_names: [...FEATURE_NAMES, ...FEATURE_NAMES.map(name => `temporal_${name}`)],
    features,
    motion_spectrum: motionSpectrum(features, input.fps),
    template,
    waveforms,
    metrics: {waveform_rms: Math.sqrt(sumSquares / (frames.length * channels * waveforms[0][0].length)),
      peak_absolute: peak, mean_band_power: meanPower},
  };
}

export type Result = Awaited<ReturnType<typeof analyze>>;

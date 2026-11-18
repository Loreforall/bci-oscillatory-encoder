import { GRID, MODES } from './cove.ts';

export const ERP_BINS = 30;
const codeSize = GRID ** 2 * MODES.length * 2;
const blockWidths = [GRID ** 2 * (3 + 3 * MODES.length), GRID ** 2 * 4 * MODES.length + 4 + 144, codeSize];
const outputSize = 8 * ERP_BINS;

type Common = { subject: string; channels: string[]; coupling: number[]; offset: number[] };
type Legacy = Common & {
  schema: 'cove-erp-v1';
  weights: number[][];
  decoder: number[][];
  decoder_offset: number[];
};
type Block = { mean: number[]; scale: number[]; anchors: number[][] };
type Advanced = Common & {
  schema: 'cove-erp-v2';
  mode: string;
  width: number;
  mixture: number[];
  blocks: (Block | null)[];
  dual: number[][];
  held_out: { rmse: number; skill: number; skill_ci95: number[] };
  test_conditions: number;
};

export type Calibration = Legacy | Advanced;
const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n) && Math.abs(n) <= 1e6;
const vector = (v: unknown, len: number): v is number[] =>
  Array.isArray(v) && v.length === len && v.every(finite);
const matrix = (v: unknown, rows: number, cols: number): v is number[][] =>
  Array.isArray(v) && v.length === rows && v.every(row => vector(row, cols));

export function parseCalibration(text: string): Calibration {
  const model = JSON.parse(text);
  const common = typeof model?.subject === 'string' && /^sub-0[12]$/.test(model.subject) &&
    Array.isArray(model.channels) && model.channels.length === 8 &&
    model.channels.every((label: unknown) => typeof label === 'string' && /^[A-Za-z0-9]{1,5}$/.test(label)) &&
    new Set(model.channels).size === 8 &&
    vector(model.coupling, 2) && model.coupling.every((v: number) => v >= 0 && v <= 12) &&
    vector(model.offset, outputSize);
  if (common && model.schema === 'cove-erp-v1' && matrix(model.weights, codeSize, outputSize) &&
      matrix(model.decoder, codeSize, 144) && vector(model.decoder_offset, 144)) return model;
  if (common && model.schema === 'cove-erp-v2' && typeof model.mode === 'string' &&
      /^[a-z_]{3,24}$/.test(model.mode) && finite(model.width) && model.width >= .01 &&
      vector(model.mixture, 3) && model.mixture.every((w: number) => w >= 0 && w <= 1) &&
      Math.abs(model.mixture.reduce((a: number, b: number) => a + b, 0) - 1) < 1e-8 &&
      Number.isInteger(model.fit_conditions) && model.fit_conditions >= 1 && model.fit_conditions <= 200 &&
      Number.isInteger(model.test_conditions) && model.test_conditions >= 1 && model.test_conditions <= 200 &&
      matrix(model.dual, model.fit_conditions, outputSize) && Array.isArray(model.blocks) &&
      model.blocks.length === 3 && model.blocks.every((block: Block | null, i: number) =>
        model.mixture[i] === 0 ? block === null :
          block && vector(block.mean, blockWidths[i]) && vector(block.scale, blockWidths[i]) &&
          block.scale.every(v => v >= .01) && matrix(block.anchors, model.fit_conditions, blockWidths[i])) &&
      finite(model.held_out?.rmse) && model.held_out.rmse >= 0 && finite(model.held_out?.skill) &&
      vector(model.held_out?.skill_ci95, 2)) return model;
  throw new Error('Select a valid COVE ERP calibration JSON exported by the research command.');
}

export function affine(x: number[], weights: number[][], offset: number[]) {
  const result = offset.slice();
  for (let i = 0; i < x.length; i++) {
    const value = x[i];
    for (let j = 0; j < result.length; j++) result[j] += value * weights[i][j];
  }
  return result;
}

export function predictErp(model: Calibration, conventional: number[], detail: number[], code: number[]) {
  if (model.schema === 'cove-erp-v1') return affine(code, model.weights, model.offset);
  const sources = [conventional, detail, code];
  const kernel = Array(model.dual.length).fill(0) as number[];
  model.blocks.forEach((block, index) => {
    if (!block) return;
    const x = sources[index].map((value, i) => (value - block.mean[i]) / block.scale[i]);
    block.anchors.forEach((anchor, n) => {
      let distance = 0;
      for (let i = 0; i < x.length; i++) distance += (x[i] - anchor[i])**2;
      kernel[n] += model.mixture[index] * Math.exp(-distance / (x.length * model.width));
    });
  });
  return affine(kernel, model.dual, model.offset);
}

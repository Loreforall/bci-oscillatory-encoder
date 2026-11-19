import { GRID, MODES } from './cove.ts';

export const SITE_COUNT = GRID ** 2;
export const MODE_COLORS = ['#176890', '#6150a0', '#087f79', '#9a4d1f'];
const width = SITE_COUNT * MODES.length * 2;

export function fieldIndex(sample: number, site: number, mode: number) {
  return sample * width + (site * MODES.length + mode) * 2;
}

export function polar(real: number, imaginary: number) {
  const amplitude = Math.hypot(real, imaginary);
  return {amplitude, phase: amplitude > 1e-9 ? Math.atan2(imaginary, real) : null};
}

export function modeTrajectory(field: Float64Array, site: number, mode: number) {
  return Array.from({length: field.length / width}, (_, sample) => {
    const index = fieldIndex(sample, site, mode);
    const real = field[index], imaginary = field[index+1];
    return {real, imaginary, ...polar(real, imaginary)};
  });
}

export function phaseDifference(target: number | null, source: number | null) {
  return target === null || source === null ? null :
    Math.atan2(Math.sin(target-source), Math.cos(target-source));
}

export function spatialMatrix(weights: {j:number;weight:number}[][]) {
  return weights.map(edges => {
    const row = Array<number>(SITE_COUNT).fill(0);
    for (const edge of edges) row[edge.j] = edge.weight;
    return row;
  });
}

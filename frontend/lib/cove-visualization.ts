import { crossMode, GRID, MODES } from './cove.ts';

export const SITE_COUNT = GRID ** 2;
export const MODE_COLORS = ['#176890', '#6150a0', '#087f79', '#9a4d1f'];

export function fieldIndex(sample: number, site: number, mode: number, count = SITE_COUNT) {
  return sample * count * MODES.length * 2 + (site * MODES.length + mode) * 2;
}

export function polar(real: number, imaginary: number) {
  const amplitude = Math.hypot(real, imaginary);
  return {amplitude, phase: amplitude > 1e-9 ? Math.atan2(imaginary, real) : null};
}

export function modeTrajectory(field: Float64Array, site: number, mode: number, count = SITE_COUNT) {
  return Array.from({length: field.length / (count * MODES.length * 2)}, (_, sample) => {
    const index = fieldIndex(sample, site, mode, count);
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
    const row = Array<number>(weights.length).fill(0);
    for (const edge of edges) row[edge.j] = edge.weight;
    return row;
  });
}

/** Select sites across the retinotopic grid, always including the current selection. */
export function rackSites(side: number, selected: number, expanded: boolean) {
  const axis = [0, 1/3, 2/3, 1].map(f => Math.round(f * (side - 1)));
  const distributed = axis.flatMap(y => axis.map(x => y * side + x));
  const overview = [0, 3, 5, 6, 9, 10, 12, 15].map(i => distributed[i]);
  return [...new Set([selected, ...(expanded ? distributed : overview)])].slice(0, expanded ? 16 : 8)
    .sort((a,b) => a-b);
}

/** Endpoint-preserving continuous display of a discrete complex transition.
 * The effective constant force includes the net spatial mixing and drive for
 * this interval. This is an interpolation of the model's rotating state. */
export function rackState(field: Float64Array, sites: number, site: number, mode: number, rate: number, time: number) {
  const sample = Math.floor(time);
  const index = fieldIndex(sample, site, mode, sites);
  const re = sample < 0 ? 0 : field[index], im = sample < 0 ? 0 : field[index + 1];
  const u = time - sample;
  if (u === 0) return {re, im};
  const low = fieldIndex(sample, site, 0, sites);
  const shift = crossMode(sample < 0 ? [0, 0] : [field[low], field[low + 1]], mode).shiftHz;
  const angle = 2 * Math.PI * (MODES[mode] + shift) / rate;
  const decay = 1 / (.09 * rate);
  const denominatorRe = Math.exp(-decay) * Math.cos(angle) - 1;
  const denominatorIm = Math.exp(-decay) * Math.sin(angle);
  const numeratorRe = Math.exp(-u * decay) * Math.cos(u * angle) - 1;
  const numeratorIm = Math.exp(-u * decay) * Math.sin(u * angle);
  const norm = denominatorRe ** 2 + denominatorIm ** 2;
  const blendRe = (numeratorRe * denominatorRe + numeratorIm * denominatorIm) / norm;
  const blendIm = (numeratorIm * denominatorRe - numeratorRe * denominatorIm) / norm;
  const dr = field[index + sites * MODES.length * 2] - re;
  const di = field[index + sites * MODES.length * 2 + 1] - im;
  return {re: re + dr * blendRe - di * blendIm, im: im + dr * blendIm + di * blendRe};
}

import { SIZE } from './features.ts';

export const GRID = 5;
export const MODES = [6, 10, 20, 40];
const axis = Array.from({length: GRID}, (_, i) => 2 * i / (GRID - 1) - 1);
export const sites = axis.flatMap(y => axis.map(x => {
  const radius = Math.hypot(y, x);
  const eccentricity = Math.expm1(Math.log(3) * radius / Math.SQRT2) / 2;
  const scale = radius ? eccentricity / radius : 0;
  return { x: (x * scale + 1) * (SIZE - 1) / 2,
    y: (y * scale + 1) * (SIZE - 1) / 2, eccentricity };
}));
const neighbors = sites.map((site, i) => sites.flatMap((other, j) =>
  Math.abs(i % GRID - j % GRID) + Math.abs(Math.floor(i / GRID) - Math.floor(j / GRID)) <= 1
    ? [{j, distance: (Math.abs(site.x - other.x) + Math.abs(site.y - other.y)) * 2 / (SIZE - 1)}] : []));

function edge(x: number) {
  return x < 0 ? -x : x >= SIZE ? 2 * SIZE - x - 2 : x;
}

function gray(rgb: Uint8Array) {
  const data = new Float64Array(SIZE * SIZE);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.round(.299 * rgb[3*i] + .587 * rgb[3*i+1] + .114 * rgb[3*i+2]) / 255;
  }
  return data;
}

function gaussian(sigma: number) {
  const radius = Math.round(4 * sigma);
  const filter = Array.from({length: 2 * radius + 1}, (_, i) => Math.exp(-.5 * ((i - radius) / sigma)**2));
  const total = filter.reduce((a, b) => a + b);
  return filter.map(value => value / total);
}

const blurLow = gaussian(1);
const blurHigh = gaussian(2);
function blur(image: Float64Array, filter: number[]) {
  const scratch = new Float64Array(image.length);
  const output = new Float64Array(image.length);
  const radius = filter.length >> 1;
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    let total = 0;
    for (let k = 0; k < filter.length; k++) total += filter[k] * image[y * SIZE + edge(x + k - radius)];
    scratch[y * SIZE + x] = total;
  }
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    let total = 0;
    for (let k = 0; k < filter.length; k++) total += filter[k] * scratch[edge(y + k - radius) * SIZE + x];
    output[y * SIZE + x] = total;
  }
  return output;
}

const kernels = [.10, .21].flatMap(frequency => [0, 1, 2, 3].map(orientation => {
  const angle = orientation * Math.PI / 4;
  return [0, -Math.PI / 2].map(phase => {
    const values = Array.from({length: 225}, (_, index) => {
      const x = index % 15 - 7;
      const y = Math.floor(index / 15) - 7;
      const u = -x * Math.cos(angle) - y * Math.sin(angle);
      return Math.exp(-(x*x + y*y) / (2 * 2.8**2)) * Math.cos(2 * Math.PI * frequency * u + phase);
    });
    const mean = values.reduce((a, b) => a + b) / values.length;
    const norm = Math.hypot(...values.map(value => value - mean));
    return values.map(value => (value - mean) / norm);
  });
}));

function corners(site: typeof sites[number]) {
  // OpenCV's remap uses 5 fractional bits for bilinear interpolation.
  const x = Math.round(32 * site.x) / 32;
  const y = Math.round(32 * site.y) / 32;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const dx = x - x0, dy = y - y0;
  return [{x: x0, y: y0, weight: (1-dx)*(1-dy)}, {x: x0+1, y: y0, weight: dx*(1-dy)},
    {x: x0, y: y0+1, weight: (1-dx)*dy}, {x: x0+1, y: y0+1, weight: dx*dy}].filter(c => c.weight);
}
const interpolation = sites.map(corners);
const taps = interpolation.map(patch => patch.map(corner => ({
  weight: corner.weight,
  indices: Array.from({length: 225}, (_, i) => edge(corner.y + Math.floor(i / 15) - 7) * SIZE
                                            + edge(corner.x + i % 15 - 7)),
})));

function sample(image: Float64Array, site: number) {
  return interpolation[site].reduce((sum, c) => sum + c.weight * image[c.y * SIZE + c.x], 0);
}

type Complex = [number, number];

function response(image: Float64Array, site: number, filter: number): Complex {
  let real = 0, imag = 0;
  const [re, im] = kernels[filter];
  for (const patch of taps[site]) {
    let pr = 0, pi = 0;
    for (let k = 0; k < re.length; k++) {
      const pixel = image[patch.indices[k]];
      pr += pixel * re[k];
      pi += pixel * im[k];
    }
    real += pr * patch.weight;
    imag += pi * patch.weight;
  }
  return [real, imag];
}

export function analyzeCove(frames: Uint8Array[], fps: number, slopes = [2, 0]) {
  if (frames.length === 1) fps = 24;
  const substeps = Math.ceil(100 / fps);
  const dt = 1 / (fps * substeps);
  const retention = Math.exp(-dt / .09);
  let previous: {brightness: number[]; low: Complex[][]} | undefined;
  let state = sites.map(() => MODES.map<Complex>(() => [0, 0]));
  const internal: number[][] = [];
  const rows = frames.map(rgb => {
    const image = gray(rgb);
    const smoothed = blur(image, blurHigh);
    const luminance = blur(image, blurLow);
    const square = blur(image.map(p => p*p), blurHigh);
    const variance = square.map((value, i) => value - smoothed[i]**2);
    const highpass = image.map((value, i) => value - smoothed[i]);
    const brightness = sites.map((_, i) => sample(luminance, i));
    const contrast = sites.map((_, i) => Math.sqrt(Math.max(0, sample(variance, i))));
    const low = sites.map((site, i) => MODES.map((_, m) => response(image, i, m)
      .map(v => v / ((.15 + contrast[i]) * (1 + site.eccentricity**2))) as Complex));
    const drive = sites.map((site, i) => MODES.map((_, m): Complex => {
      const [hr, hi] = response(highpass, i, m + 4);
      const attenuation = (.15 + contrast[i]) * (1 + 5 * site.eccentricity**2);
      const [lr, li] = low[i][m];
      const [pr, pi] = previous?.low[i][m] ?? [0, 0];
      const temporal = (li*pr - lr*pi) / (.1 + Math.hypot(lr, li)*Math.hypot(pr, pi));
      const motion = previous ? Math.tanh((brightness[i] - previous.brightness[i]) * fps / 12) : 0;
      const re = .5*lr + .25*hr/attenuation + .15*(brightness[i] - .5) + .1*motion;
      const im = .5*li + .25*hi/attenuation + .1*temporal;
      const gain = 1 + Math.hypot(re, im);
      return [re/gain, im/gain];
    }));
    previous = {brightness, low};
    const weights = neighbors.map((edges, i) => {
      const unscaled = edges.map(({j, distance}) => Math.exp(-slopes[0]*distance -
        slopes[1]*Math.abs(contrast[i] - contrast[j])));
      const norm = unscaled.reduce((a,b) => a+b);
      return edges.map(({j}, k) => ({j, weight: unscaled[k]/norm}));
    });
    const mix = (values: Complex[][], i: number, m: number): Complex => weights[i].reduce(
      ([re, im], edge) => [re + edge.weight*values[edge.j][m][0], im + edge.weight*values[edge.j][m][1]],
      [0, 0] as Complex);
    const input = drive.map((modes, i) => modes.map(([re, im], m) => {
      const [mr, mi] = mix(drive, i, m);
      return [.8*re + .2*mr, .8*im + .2*mi];
    }));
    for (let step = 0; step < substeps * (frames.length === 1 ? 5 : 1); step++) {
      state = sites.map((_, i) => MODES.map((hz, m): Complex => {
        const angle = dt * (2*Math.PI*hz + 2.4*state[i][0][1]*m);
        const [re, im] = mix(state, i, m);
        return [retention*(re*Math.cos(angle) - im*Math.sin(angle)) + (1-retention)*input[i][m][0],
          retention*(re*Math.sin(angle) + im*Math.cos(angle)) + (1-retention)*input[i][m][1]];
      }));
      internal.push(state.map(modes => modes.reduce((sum, [re]) => sum + re, 0)));
    }
    return {brightness, contrast, drive, weights, state};
  });
  return {rows, internal, internalRate: fps * substeps};
}

export function spectrogram(signal: number[], hz: number) {
  const size = Math.min(signal.length, Math.round(.25 * hz));
  if (size < 4) return [];
  const hop = Math.max(1, Math.round(size / 2));
  return Array.from({length: 1 + Math.floor((signal.length - size) / hop)}, (_, column) =>
    Array.from({length: (size >> 1) + 1}, (_, bin) => {
      let re = 0, im = 0;
      for (let sample = 0; sample < size; sample++) {
        const angle = 2 * Math.PI * bin * sample / size;
        const value = signal[column * hop + sample] * (.5 - .5*Math.cos(2*Math.PI*sample/(size - 1)));
        re += value * Math.cos(angle);
        im -= value * Math.sin(angle);
      }
      return (re*re + im*im) / size**2;
    }));
}

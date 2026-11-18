import { SIZE, spatialPower } from './features.ts';

export const GRID = 5;
export const HAAR_SCALES = [1, 2, 4];
export const FOURIER_EDGES = [.0625, .125, .25];
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
    data[i] = ((9798*rgb[3*i] + 19235*rgb[3*i+1] + 3735*rgb[3*i+2] + 16384) >> 15) / 255;
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

const pixelTaps = Array.from({length: 12}, (_, pixel) => {
  const from = pixel * SIZE / 12, to = (pixel + 1) * SIZE / 12;
  return Array.from({length: Math.ceil(to) - Math.floor(from)}, (_, i) => {
    const index = Math.floor(from) + i;
    return {index, weight: (Math.min(to, index + 1) - Math.max(from, index)) * 12 / SIZE};
  });
});

function areaPixels(image: Float64Array) {
  return pixelTaps.flatMap(y => pixelTaps.map(x => y.reduce((total, row) =>
    total + row.weight * x.reduce((sum, col) => sum + col.weight*image[row.index*SIZE + col.index], 0), 0)));
}

function bandRms(power: Float64Array) {
  const energy = [0, 0, 0, 0];
  power.forEach((value, i) => {
    const x = i % SIZE, y = Math.floor(i / SIZE);
    const radius = Math.hypot(Math.min(x, SIZE-x), Math.min(y, SIZE-y)) / SIZE;
    const band = FOURIER_EDGES.filter(bound => radius >= bound).length;
    energy[band] += value;
  });
  return energy.map(value => Math.sqrt(value) / SIZE**2);
}

export function flattenCode(state: Complex[][]) {
  return state.flatMap(modes => [...modes.map(pair => pair[0]), ...modes.map(pair => pair[1])]);
}

export function analyzeCove(frames: Uint8Array[], fps: number, slopes = [2, 0], advanced = false) {
  if (frames.length === 1) fps = 24;
  const substeps = Math.ceil(100 / fps);
  const dt = 1 / (fps * substeps);
  const retention = Math.exp(-dt / .09);
  let previous: {brightness: number[]; low: Complex[][]} | undefined;
  let state = sites.map(() => MODES.map<Complex>(() => [0, 0]));
  const internal: number[][] = [];
  const history: number[][] = [];
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
    history.push(brightness);
    const index = history.length - 1;
    const wavelets = sites.map((_, i) => HAAR_SCALES.map(size => {
      let difference = 0;
      for (let lag = 0; lag < size; lag++) {
        difference += history[Math.max(0, index-lag)][i] - history[Math.max(0, index-lag-size)][i];
      }
      return difference / (Math.SQRT2 * size);
    }));
    const high = sites.map((site, i) => MODES.map((_, m) => response(highpass, i, m + 4)
      .map(v => v / ((.15 + contrast[i]) * (1 + 5*site.eccentricity**2))) as Complex));
    const motion = brightness.map((value, i) => previous ? Math.tanh((value - previous.brightness[i]) * fps / 12) : 0);
    const phaseChange = low.map((modes, i) => modes.map(([lr, li], m) => {
      const [pr, pi] = previous?.low[i][m] ?? [0, 0];
      return (li*pr - lr*pi) / (.1 + Math.hypot(lr, li)*Math.hypot(pr, pi));
    }));
    const drive = sites.map((site, i) => MODES.map((_, m): Complex => {
      const [hr, hi] = high[i][m];
      const [lr, li] = low[i][m];
      const re = .5*lr + .25*hr + .15*(brightness[i] - .5) + .1*motion[i];
      const im = .5*li + .25*hi + .1*phaseChange[i][m];
      const gain = 1 + Math.hypot(re, im);
      const [fast, medium, slow] = wavelets[i];
      const correction = advanced ? .25 * [slow, medium, fast, fast-medium][m] : 0;
      const r = re/gain, q = im/gain + correction;
      const bounded = Math.max(1, Math.hypot(r, q));
      return [r/bounded, q/bounded];
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
        const gate = advanced && m > 0 ? 1 - .18*(1-state[i][0][0])/2 : 1;
        return [retention*(re*Math.cos(angle) - im*Math.sin(angle)) + (1-retention)*gate*input[i][m][0],
          retention*(re*Math.sin(angle) + im*Math.cos(angle)) + (1-retention)*gate*input[i][m][1]];
      }));
      internal.push(state.map(modes => modes.reduce((sum, [re]) => sum + re, 0)));
    }
    const power = spatialPower(image);
    const spectrum = bandRms(power);
    const pixels = areaPixels(image);
    const conventional = sites.flatMap((_, i) => [brightness[i], contrast[i], motion[i], ...phaseChange[i],
      ...low[i].map(pair => Math.hypot(...pair)), ...high[i].map(pair => Math.hypot(...pair))]);
    const detail = sites.flatMap((_, i) => low[i].flatMap(([re, im], m) => [re, im, ...high[i][m]]));
    return {brightness, contrast, drive, weights, state, spectrum, pixels, conventional, detail, wavelets};
  });
  return {rows, internal, internalRate: fps * substeps};
}

export function spectrumWindowLength(length: number, hz: number) {
  return Math.min(length, Math.round(.25 * hz));
}

export function spectrogram(signal: number[], hz: number) {
  const size = spectrumWindowLength(signal.length, hz);
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

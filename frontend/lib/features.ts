export const SIZE = 64;
const PIXELS = SIZE ** 2;
const TAU = 2 * Math.PI;

export const FEATURE_NAMES = [
  'brightness', 'contrast', 'edge_density', 'low_frequency', 'high_frequency',
  'foveal', 'peripheral', 'motion',
];

const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;

function fft(real: Float64Array, imaginary: Float64Array) {
  for (let i = 1, reversed = 0; i < SIZE; i++) {
    let bit = SIZE >> 1;
    while (reversed & bit) {
      reversed ^= bit;
      bit >>= 1;
    }
    reversed ^= bit;
    if (i < reversed) {
      [real[i], real[reversed]] = [real[reversed], real[i]];
      [imaginary[i], imaginary[reversed]] = [imaginary[reversed], imaginary[i]];
    }
  }
  for (let width = 2; width <= SIZE; width *= 2) {
    const half = width / 2;
    for (let offset = 0; offset < half; offset++) {
      const cosine = Math.cos(-TAU * offset / width);
      const sine = Math.sin(-TAU * offset / width);
      for (let i = offset; i < SIZE; i += width) {
        const j = i + half;
        const r = cosine * real[j] - sine * imaginary[j];
        const im = sine * real[j] + cosine * imaginary[j];
        real[j] = real[i] - r;
        imaginary[j] = imaginary[i] - im;
        real[i] += r;
        imaginary[i] += im;
      }
    }
  }
}

function spatialEnergy(gray: Uint8Array) {
  const real = Float64Array.from(gray, value => value / 255);
  const imaginary = new Float64Array(PIXELS);
  for (let y = 0; y < SIZE; y++) {
    const start = y * SIZE;
    fft(real.subarray(start, start + SIZE), imaginary.subarray(start, start + SIZE));
  }
  const columnReal = new Float64Array(SIZE);
  const columnImaginary = new Float64Array(SIZE);
  let low = 0;
  let high = 0;
  for (let x = 0; x < SIZE; x++) {
    for (let y = 0; y < SIZE; y++) {
      columnReal[y] = real[y * SIZE + x];
      columnImaginary[y] = imaginary[y * SIZE + x];
    }
    fft(columnReal, columnImaginary);
    for (let y = 0; y < SIZE; y++) {
      if (x === 0 && y === 0) continue;
      const radius = Math.hypot(Math.min(x, SIZE - x), Math.min(y, SIZE - y)) / (SIZE / 2);
      const power = columnReal[y] ** 2 + columnImaginary[y] ** 2;
      if (radius <= 0.35) low += power;
      else high += power;
    }
  }
  const total = low + high;
  return total < 1e-12 ? [0, 0] : [low / total, high / total];
}

function cannyDensity(gray: Uint8Array) {
  const stride = SIZE + 2;
  const magnitude = new Int16Array(stride ** 2);
  const dx = new Int16Array(stride ** 2);
  const dy = new Int16Array(stride ** 2);
  // Sobel uses replicated image borders. Nonmaximum suppression sees zero outside.
  for (let y = 0; y < SIZE; y++) {
    const top = Math.max(0, y - 1) * SIZE;
    const middle = y * SIZE;
    const bottom = Math.min(SIZE - 1, y + 1) * SIZE;
    for (let x = 0; x < SIZE; x++) {
      const left = Math.max(0, x - 1);
      const right = Math.min(SIZE - 1, x + 1);
      const i = (y + 1) * stride + x + 1;
      dx[i] = gray[top + right] - gray[top + left] + 2 * (gray[middle + right] - gray[middle + left])
        + gray[bottom + right] - gray[bottom + left];
      dy[i] = gray[bottom + left] + 2 * gray[bottom + x] + gray[bottom + right]
        - gray[top + left] - 2 * gray[top + x] - gray[top + right];
      magnitude[i] = Math.abs(dx[i]) + Math.abs(dy[i]);
    }
  }
  const candidates = new Uint8Array(stride ** 2);
  const strong: number[] = [];
  for (let y = 1; y <= SIZE; y++) {
    for (let x = 1; x <= SIZE; x++) {
      const i = y * stride + x;
      const strength = magnitude[i];
      if (strength <= 50) continue;
      const horizontal = Math.abs(dx[i]);
      const vertical = Math.abs(dy[i]) * 32768;
      // Fixed-point tan(22.5°) and tan(67.5°) match OpenCV's direction bins.
      const shallow = horizontal * 13573;
      const diagonal = vertical >= shallow && vertical <= shallow + horizontal * 65536;
      const offset = vertical < shallow ? 1 : !diagonal ? stride
        : stride + ((dx[i] ^ dy[i]) < 0 ? -1 : 1);
      if (strength <= magnitude[i - offset] || strength < magnitude[i + offset]
          || (diagonal && strength === magnitude[i + offset])) continue;
      if (strength > 150) strong.push(i);
      else candidates[i] = 1;
    }
  }
  const neighbors = [-stride - 1, -stride, -stride + 1, -1, 1, stride - 1, stride, stride + 1];
  for (let i = 0; i < strong.length; i++) {
    for (const offset of neighbors) {
      const neighbor = strong[i] + offset;
      if (candidates[neighbor]) {
        candidates[neighbor] = 0;
        strong.push(neighbor);
      }
    }
  }
  return strong.length / PIXELS;
}

const kernels = [1, 2, 3].map(scale => {
  const weights = Array.from({length: 8 * scale + 1}, (_, i) => {
    const t = i / scale - 4;
    return Math.cos(5 * t) * Math.exp(-t * t / 2);
  });
  const center = mean(weights);
  const norm = weights.reduce((sum, value) => sum + Math.abs(value - center), 0);
  return weights.map(value => (value - center) / norm);
});

export function extractFeatures(frames: Uint8Array[]) {
  let previous: Uint8Array | undefined;
  const spatial = frames.map(rgb => {
    if (rgb.length !== PIXELS * 3) throw new Error('Expected packed 64 × 64 RGB frames.');
    const gray = new Uint8Array(PIXELS);
    let total = 0;
    let center = 0;
    let motion = 0;
    for (let i = 0; i < PIXELS; i++) {
      // OpenCV's integer RGB-to-gray coefficients and rounding.
      const j = i * 3;
      const value = (9798 * rgb[j] + 19235 * rgb[j + 1] + 3735 * rgb[j + 2] + 16384) >> 15;
      gray[i] = value;
      total += value;
      const x = i % SIZE;
      const y = Math.floor(i / SIZE);
      if (x >= 22 && x < 42 && y >= 22 && y < 42) center += value;
      if (previous) motion += Math.abs(value - previous[i]);
    }
    previous = gray;
    const brightness = total / PIXELS;
    const variance = gray.reduce((sum, value) => sum + (value - brightness) ** 2, 0) / PIXELS;
    return [brightness / 255, Math.sqrt(variance) / 255, cannyDensity(gray), ...spatialEnergy(gray),
      center / (400 * 255), (total - center) / ((PIXELS - 400) * 255), motion / (PIXELS * 255)];
  });
  return spatial.map((row, frame) => row.concat(row.map((_, feature) => {
    if (frames.length === 1) return 0;
    return mean(kernels.map(kernel => Math.abs(kernel.reduce((sum, weight, i) => {
      const index = Math.max(0, Math.min(frame + i - (kernel.length >> 1), frames.length - 1));
      return sum + weight * spatial[index][feature];
    }, 0))));
  })));
}

export function motionSpectrum(features: number[][], fps: number) {
  const motion = features.map(row => row[7]);
  const center = mean(motion);
  const length = motion.length;
  const bins = Array.from({length: Math.floor(length / 2) + 1}, (_, bin) => bin);
  return {
    hz: bins.map(bin => bin * fps / length),
    power: bins.map(bin => {
      let real = 0;
      let imaginary = 0;
      motion.forEach((value, i) => {
        const angle = TAU * bin * i / length;
        real += (value - center) * Math.cos(angle);
        imaginary -= (value - center) * Math.sin(angle);
      });
      return (real ** 2 + imaginary ** 2) / length ** 2;
    }),
  };
}

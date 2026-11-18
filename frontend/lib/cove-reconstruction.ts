import { affine } from './cove-calibration.ts';
import { GRID, MODES } from './cove.ts';

type Edge = {j: number; weight: number};
type Renderer = {weights: number[][]; offset: number[]; rate: number};
const coefficientCount = 2 * MODES.length;

export function refineCode(code: number[][], pixels: number[][], graph: Edge[][][], renderer: Renderer,
  steps = 28) {
  let state = code.map(row => row.slice());
  for (let step = 0; step < steps; step++) {
    const change = state.map((row, t) => row.map((value, i) => value - code[t][i]));
    state = state.map((row, t) => {
      const residual = affine(row, renderer.weights, renderer.offset).map((value, i) => value - pixels[t][i]);
      const spatial = change[t].slice();
      graph[t].forEach((edges, i) => {
        for (const {j, weight} of edges) {
          for (let k = 0; k < coefficientCount; k++) spatial[i*coefficientCount + k] -= weight * change[t][j*coefficientCount + k];
        }
      });
      const gradient = renderer.weights.map((weights, i) => {
        let error = 0;
        for (let k = 0; k < weights.length; k++) error += weights[k]*residual[k];
        let dt = 0;
        if (t > 0) dt += change[t][i] - change[t-1][i];
        if (t + 1 < code.length) dt += change[t][i] - change[t+1][i];
        return 2 * error / residual.length + .12 * change[t][i] + .04 * dt + .07 * spatial[i];
      });
      graph[t].forEach((edges, i) => {
        for (const {j, weight} of edges) {
          for (let k = 0; k < coefficientCount; k++) gradient[j*coefficientCount + k] -= .07 * weight * spatial[i*coefficientCount + k];
        }
      });
      const next = code[t].slice();
      for (let i = 0; i < GRID**2; i++) for (let b = 0; b < MODES.length; b++) {
        const r = i*coefficientCount + b, q = r + MODES.length;
        let dr = change[t][r] - renderer.rate*gradient[r];
        let dq = change[t][q] - renderer.rate*gradient[q];
        const divisor = Math.max(1, Math.hypot(dr, dq) / .15);
        dr /= divisor;
        dq /= divisor;
        if (Math.hypot(next[r]+dr, next[q]+dq) > 1) {
          const a = dr*dr + dq*dq;
          const b = next[r]*dr + next[q]*dq;
          const c = next[r]**2 + next[q]**2 - 1;
          const scale = (-b + Math.sqrt(Math.max(0, b*b-a*c))) / a;
          dr *= scale;
          dq *= scale;
        }
        next[r] += dr;
        next[q] += dq;
      }
      return next;
    });
  }
  return state;
}

export function imageError(actual: number[], estimate: number[]) {
  const mse = actual.reduce((sum, value, i) => sum + (value - estimate[i])**2, 0) / actual.length;
  return {rmse: Math.sqrt(mse), psnr: mse === 0 ? Infinity : -10*Math.log10(mse)};
}

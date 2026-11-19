import type { Input } from './encoder.ts';
import { SIZE } from './features.ts';

export const exampleInput: Input = {
  kind: 'synthetic', width: SIZE, height: SIZE, fps: 24, sampling: 'cove-moving-grating',
};

export function coveExample() {
  return Array.from({length: 48}, (_, frame) => {
    const rgb = new Uint8Array(SIZE * SIZE * 3);
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const u = x/(SIZE-1) - .5, v = y/(SIZE-1) - .5;
      const envelope = Math.exp(-3.4*(u*u+v*v));
      const wave = .5 + .42 * envelope * Math.sin(2*Math.PI*(3.8*u+1.8*v) - frame*Math.PI/12);
      const index = (y*SIZE+x)*3;
      rgb[index] = Math.round(14 + 210*wave);
      rgb[index+1] = Math.round(45 + 197*wave);
      rgb[index+2] = Math.round(89 + 159*wave);
    }
    return rgb;
  });
}

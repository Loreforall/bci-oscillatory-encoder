import { readFileSync } from 'node:fs';
import { corticalResponse } from '../frontend/lib/cortical.ts';

const {frames} = JSON.parse(readFileSync(0, 'utf8'));
const output = frames.map(row => {
  const {field, rate} = corticalResponse([row], 120, 12, {duration:.6,stimulusDuration:.1});
  const width = row.contrast.length * 8;
  // Read out the signed E and I sample means in physiological onset bins.
  return Array.from({length:30},(_,bin)=>{
    const start=Math.floor(bin*.02*rate), end=Math.floor((bin+1)*.02*rate);
    return Array.from({length:width},(_,j)=>{
      let sum=0;
      for(let t=start;t<end;t++) sum+=field[t*width+j];
      return sum/(end-start);
    });
  }).flat();
});
process.stdout.write(JSON.stringify(output));

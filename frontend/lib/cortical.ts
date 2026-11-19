/** Exploratory signed population deviations, in arbitrary model units. */
export const CORTICAL_BANDS = [8, 14, 24, 38];
export const CORTICAL_SUBSTEPS = 4;
export const CORTICAL_LATENCY = .018;

export type CorticalFrame = {
  detail: number[];
  contrast: number[];
  motion: number[];
  weights: {j:number;weight:number}[][];
};

export function corticalResponse(rows: CorticalFrame[], referenceRate: number, referenceSteps: number,
  {duration = rows.length * referenceSteps / referenceRate, stimulusDuration = duration, lateral = true}: {duration?:number;stimulusDuration?:number;lateral?:boolean} = {}) {
  const rate = referenceRate * CORTICAL_SUBSTEPS;
  const steps = referenceSteps * CORTICAL_SUBSTEPS;
  const count = Math.round(duration * rate), sites = rows[0].contrast.length;
  const size = sites * CORTICAL_BANDS.length;
  const field = new Float64Array(count * size * 2);
  let e = new Float64Array(size), i = new Float64Array(size);
  let nextE = new Float64Array(size), nextI = new Float64Array(size);
  const input = new Float64Array(size), adaptationFast = new Float64Array(size), adaptationSlow = new Float64Array(size);
  const gains = [.014, .065, .18].map(tau => -Math.expm1(-1/(rate*tau)));
  const diffusion = lateral ? -Math.expm1(-4/rate) : 0;
  const prepared = rows.map(row => {
    const drive = new Float64Array(size);
    const operators = new Float64Array(size * 4);
    for (let s=0; s<sites; s++) for (let m=0; m<CORTICAL_BANDS.length; m++) {
      const p=s*CORTICAL_BANDS.length+m, j=p*4;
      const low=Math.hypot(row.detail[j],row.detail[j+1]);
      const high=Math.hypot(row.detail[j+2],row.detail[j+3]);
      const contrast=row.contrast[s];
      drive[p]=Math.tanh(.62*low+.38*high+.45*contrast+.2*Math.abs(row.motion[s]));
      const omega=2*Math.PI*CORTICAL_BANDS[m]*(1+.12*Math.tanh(contrast*3));
      const damping=11+20*contrast;
      const a=Math.exp(-damping/rate)*Math.cos(omega/rate);
      const b=Math.exp(-damping/rate)*Math.sin(omega/rate);
      const norm=damping*damping+omega*omega;
      operators.set([a,b,((1-a)*damping+b*omega)/norm,((1-a)*omega-b*damping)/norm],j);
    }
    return {drive,operators,weights:row.weights};
  });
  // Delay applies to each change in stimulus, including the experimental offset.
  const latencySamples = Math.round(CORTICAL_LATENCY * rate);
  for (let t=latencySamples; t<count; t++) {
    const stimulusSample=t-latencySamples;
    const active=stimulusSample < Math.round(stimulusDuration*rate);
    const row=prepared[Math.min(prepared.length-1,Math.floor(stimulusSample/steps))];
    const offset=t*size*2;
    for (let s=0; s<sites; s++) for (let m=0; m<CORTICAL_BANDS.length; m++) {
      const p=s*CORTICAL_BANDS.length+m, j=p*4;
      const drive=active?row.drive[p]:0;
      input[p]+=gains[0]*(drive-input[p]);
      adaptationFast[p]+=gains[1]*(input[p]-adaptationFast[p]);
      adaptationSlow[p]+=gains[2]*(input[p]-adaptationSlow[p]);
      const force=120*(.8*drive+.55*input[p]-.62*adaptationFast[p]-.36*adaptationSlow[p]);
      let mixedE=e[p], mixedI=i[p];
      if (diffusion) for (const edge of row.weights[s]) {
        const q=edge.j*CORTICAL_BANDS.length+m;
        const share=diffusion*edge.weight;
        mixedE+=share*(e[q]-e[p]);
        mixedI+=share*(i[q]-i[p]);
      }
      const a=row.operators[j], b=row.operators[j+1];
      nextE[p]=a*mixedE-b*mixedI+row.operators[j+2]*force;
      nextI[p]=b*mixedE+a*mixedI+row.operators[j+3]*force;
      field[offset+2*p]=nextE[p];
      field[offset+2*p+1]=nextI[p];
    }
    [e,nextE]=[nextE,e];
    [i,nextI]=[nextI,i];
  }
  return {field, rate, count, latencySamples};
}

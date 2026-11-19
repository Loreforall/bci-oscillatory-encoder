import { CORTICAL_BANDS, CORTICAL_LATENCY, type CorticalFrame } from './cortical.ts';

export const RECORDING_RATE = 480;
export const RECORDING_BASELINE = .5;
export const RECORDING_SEED = 1731;

function gaussian(seed:number) {
  let state=seed>>>0 || 1;
  const uniform=()=>{
    state^=state<<13; state^=state>>>17; state^=state<<5;
    return ((state>>>0)+.5)/4294967296;
  };
  return ()=>Math.sqrt(-2*Math.log(uniform()))*Math.cos(2*Math.PI*uniform());
}

/** Retinotopic current proxy with common-average reference, in model units. */
export function corticalRecording(rows:CorticalFrame[], fps:number, seed=RECORDING_SEED, still=rows.length===1) {
  const rate=RECORDING_RATE, onset=Math.round(RECORDING_BASELINE*rate);
  const count=onset+Math.round((still?2.5:rows.length/fps)*rate);
  const delay=Math.round(CORTICAL_LATENCY*rate), sites=rows[0].contrast.length;
  const bands=CORTICAL_BANDS.length, size=sites*bands;
  const field=new Float64Array(count*sites);
  const normal=gaussian(seed);
  let e=new Float64Array(size), i=new Float64Array(size);
  let nextE=new Float64Array(size), nextI=new Float64Array(size);
  const colored=new Float64Array(size), shared=new Float64Array(bands), adaptation=new Float64Array(size);
  const slow=new Float64Array(sites), filtered=new Float64Array(sites);
  const tau=[.045,.027,.014,.007];
  const rho=tau.map(time=>Math.exp(-1/(time*rate)));
  const innovation=rho.map(a=>Math.sqrt(1-a*a));
  const adapt=-Math.expm1(-1/(.24*rate)), lateral=-Math.expm1(-9/rate);
  const hp=-Math.expm1(-2*Math.PI*.7/rate), lp=-Math.expm1(-2*Math.PI*85/rate);
  const neutral=rows[0].weights.map(edges=>edges.map(({j})=>({j,weight:1/edges.length})));
  const prepare=(row?:CorticalFrame)=>{
    const drive=new Float64Array(size), rotation=new Float64Array(size*4);
    for(let q=0;q<size;q++){
      const s=Math.floor(q/bands), m=q%bands, k=q*4;
      const contrast=row?.contrast[s]??0;
      const low=row?Math.hypot(row.detail[k],row.detail[k+1]):0;
      const high=row?Math.hypot(row.detail[k+2],row.detail[k+3]):0;
      const d=Math.tanh(.65*low+.4*high+.65*contrast+.18*Math.abs(row?.motion[s]??0));
      drive[q]=d;
      const alpha=9+5*m+9*d, omega=2*Math.PI*CORTICAL_BANDS[m]*(1+.16*d);
      const a=Math.exp(-alpha/rate)*Math.cos(omega/rate), b=Math.exp(-alpha/rate)*Math.sin(omega/rate);
      const norm=alpha*alpha+omega*omega;
      rotation.set([a,b,((1-a)*alpha+b*omega)/norm,((1-a)*omega-b*alpha)/norm],k);
    }
    return {drive,rotation,weights:row?.weights??neutral};
  };
  const baseline=prepare(), stimuli=rows.map(prepare);
  for(let t=-rate;t<count;t++){
    const elapsed=t-onset-delay;
    const frame=Math.min(rows.length-1,Math.floor(elapsed*fps/rate));
    const {drive,rotation,weights}=elapsed<0?baseline:stimuli[frame];
    for(let m=0;m<bands;m++) shared[m]=rho[m]*shared[m]+innovation[m]*normal();
    let mean=0;
    for(let s=0;s<sites;s++){
      let current=0, localSlow=0;
      for(let m=0;m<bands;m++) localSlow+=e[s*bands+m];
      for(let m=0;m<bands;m++){
        const q=s*bands+m, k=q*4, d=drive[q];
        colored[q]=rho[m]*colored[q]+innovation[m]*normal();
        adaptation[q]+=adapt*(d-adaptation[q]);
        let u=e[q], v=i[q];
        for(const edge of weights[s]){
          const n=edge.j*bands+m, mix=lateral*edge.weight;
          u+=mix*(e[n]-e[q]); v+=mix*(i[n]-i[q]);
        }
        const background=.9*colored[q]+.55*shared[m];
        const gain=1+1.45*d+.2*Math.tanh(localSlow/(bands*.3));
        const force=34*gain*background+100*(d-.58*adaptation[q]);
        nextE[q]=rotation[k]*u-rotation[k+1]*v+rotation[k+2]*force;
        nextI[q]=rotation[k+1]*u+rotation[k]*v+rotation[k+3]*force;
        current+=(nextE[q]-.65*nextI[q])*(1+.25*m);
      }
      current/=bands;
      slow[s]+=hp*(current-slow[s]);
      filtered[s]+=lp*(current-slow[s]-filtered[s]);
      mean+=filtered[s];
    }
    if(t>=0) for(let s=0;s<sites;s++) field[t*sites+s]=filtered[s]-mean/sites;
    [e,nextE]=[nextE,e]; [i,nextI]=[nextI,i];
  }
  return {field,rate,count,onset,sites};
}

export function recordingSites(count:number, selected:number, expanded:boolean) {
  const limit=Math.min(count,expanded?64:32);
  const candidates=Array.from({length:limit},(_,i)=>Math.round(i*(count-1)/Math.max(1,limit-1)));
  if(!candidates.includes(selected)){
    const closest=candidates.reduce((best,v,i)=>Math.abs(v-selected)<Math.abs(candidates[best]-selected)?i:best,0);
    candidates[closest]=selected;
  }
  return candidates;
}

export function recordingSpectra(field:Float64Array, sites:number, site:number, comparison:number, from:number, end:number, rate:number) {
  const n=256, hop=128, bins=49;
  const frames: number[][]=[];
  const power=Array<number>(bins).fill(0), other=power.slice(), real=power.slice(), imaginary=power.slice();
  const window=Array.from({length:n},(_,t)=>.5-.5*Math.cos(2*Math.PI*t/(n-1)));
  const scale=2/(rate*window.reduce((sum,w)=>sum+w*w,0));
  for(let start=from;start+n<=end;start+=hop){
    const row=[];
    for(let k=0;k<bins;k++){
      let ar=0,ai=0,br=0,bi=0;
      for(let t=0;t<n;t++){
        const angle=2*Math.PI*k*t/n, re=Math.cos(angle)*window[t], im=-Math.sin(angle)*window[t];
        const idx=(start+t)*sites;
        ar+=field[idx+site]*re; ai+=field[idx+site]*im;
        br+=field[idx+comparison]*re; bi+=field[idx+comparison]*im;
      }
      const p=ar*ar+ai*ai, q=br*br+bi*bi;
      row.push(10*Math.log10(Math.max(1e-12,p*scale*(k===0?.5:1))));
      power[k]+=p; other[k]+=q; real[k]+=ar*br+ai*bi; imaginary[k]+=ai*br-ar*bi;
    }
    frames.push(row);
  }
  return {frames, hz:Array.from({length:bins},(_,k)=>k*rate/n), n, hop,
    psd:power.map((p,k)=>10*Math.log10(Math.max(1e-12,p*scale*(k===0?.5:1)/Math.max(1,frames.length)))),
    coherence:power.map((p,k)=>frames.length<2?null:Math.min(1,(real[k]**2+imaginary[k]**2)/Math.max(1e-30,p*other[k])))};
}

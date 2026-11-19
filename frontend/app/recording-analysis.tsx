'use client';

import { memo, useMemo } from 'react';
import { type corticalRecording, recordingSites } from '../lib/cortical-recording';
import { recordingSpectra } from '../lib/recording-spectra';
import { ScientificPlot } from './cove-science';

export default memo(function RecordingAnalysis({recording,site}:{
  recording:ReturnType<typeof corticalRecording>;site:number;
}) {
  const {field,sites,rate,count,onset}=recording;
  const comparison=site===Math.floor(sites/2)?0:Math.floor(sites/2);
  const end=Math.min(count,onset+3*rate);
  const {frames,hz,psd,coherence,n,hop}=useMemo(()=>recordingSpectra(field,sites,site,comparison,0,end,rate),
    [field,sites,site,comparison,end,rate]);
  const {channels,correlations}=useMemo(()=>{
    const channels=recordingSites(sites,site,false);
    const vectors=channels.map(channel=>{
      const values=Array.from({length:end},(_,t)=>field[t*sites+channel]);
      const mean=values.reduce((a,b)=>a+b,0)/end;
      const centered=values.map(v=>v-mean);
      const norm=Math.sqrt(centered.reduce((a,b)=>a+b*b,0));
      return centered.map(v=>v/Math.max(norm,1e-12));
    });
    return {channels,correlations:vectors.map(a=>vectors.map(b=>a.reduce((sum,v,t)=>sum+v*b[t],0)))};
  },[field,sites,site,end]);
  const color=(v:number)=>v>=0?`hsl(207 56% ${97-55*v}%)`:`hsl(24 62% ${97+55*v}%)`;
  const peak=Math.max(-120,...frames.flat()), floor=peak-55;
  return <section className="science-stack">
    <h2>Simulated LFP proxy analysis</h2>
    <p>These views use the same common-referenced, 480 Hz samples as the waveform rack. Analysis covers the baseline and up to the first 3 seconds after onset. Site labels refer to visual locations.</p>
    <div className="science-columns">
      <ScientificPlot title={`Welch power spectrum · S${site+1}`} unit="dB re 1 a.u.²/Hz" series={[{name:'Power density',color:'#233f58',data:psd}]} range={[floor,peak+5]} rate={1/(hz[1]-hz[0])} offset={0} xUnit="Hz" domain={[0,hz.length-1]}/>
      <ScientificPlot title={`Magnitude-squared coherence · S${site+1} / S${comparison+1}`} unit="0–1" range={[0,1]} series={[{name:'Coherence',color:'#233f58',data:coherence}]} rate={1/(hz[1]-hz[0])} offset={0} xUnit="Hz" domain={[0,hz.length-1]}/>
      <figure className="cove-figure"><figcaption>Windowed power · S{site+1}<span>Hz · dB/Hz</span></figcaption>
        <svg viewBox="0 0 420 188" role="img" aria-label="Spectrogram of the selected simulated site">
          {frames.map((row,i)=>row.map((v,k)=><rect key={`${i}:${k}`} x={38+i*365/frames.length} y={12+(row.length-1-k)*150/row.length}
            width={365/frames.length+.2} height={150/row.length+.2} fill={color(Math.max(0,(v-floor)/55))}>
            <title>{((i*hop+n/2-onset)/rate).toFixed(3)} s · {hz[k].toFixed(1)} Hz · {v.toFixed(1)} dB/Hz</title>
          </rect>))}
          {[0,24,48].map(k=><text key={k} x="32" y={16+(48-k)*150/49} textAnchor="end">{Math.round(hz[k])}</text>)}
          {frames.length>0&&<><text x="38" y="180">{((n/2-onset)/rate).toFixed(2)} s</text><text x="402" y="180" textAnchor="end">{(((frames.length-1)*hop+n/2-onset)/rate).toFixed(2)} s</text></>}
        </svg>
        <div className="science-scale"><span>{floor.toFixed(0)} dB/Hz</span><i style={{background:'linear-gradient(90deg,hsl(207 56% 97%),hsl(207 56% 42%))'}}/><span>{peak.toFixed(0)} dB/Hz</span></div>
      </figure>
      <figure className="cove-figure"><figcaption>Channel correlation<span>Pearson r · −1 to +1</span></figcaption>
        <svg viewBox="0 0 355 355" role="img" aria-label="Correlation between waveform rack sites">
          {correlations.map((row,i)=>row.map((v,j)=><rect key={`${i}:${j}`} x={28+j*310/channels.length} y={25+i*310/channels.length}
            width={310/channels.length+.15} height={310/channels.length+.15} fill={color(v)}>
            <title>S{channels[i]+1} / S{channels[j]+1} · r {v.toFixed(3)}</title>
          </rect>))}
          {channels.map((s,i)=>i%8===0&&<g key={s}><text x="25" y={34+i*310/channels.length} textAnchor="end">S{s+1}</text><text x={28+i*310/channels.length} y="18">S{s+1}</text></g>)}
        </svg>
        <div className="science-scale"><span>−1</span><i style={{background:'linear-gradient(90deg,hsl(24 62% 42%), #f3f8fb,hsl(207 56% 42%))'}}/><span>+1</span></div>
      </figure>
    </div>
  </section>;
});

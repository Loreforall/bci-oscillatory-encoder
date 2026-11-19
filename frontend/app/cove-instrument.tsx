'use client';

import { memo, useMemo, useState } from 'react';
import { MODES, FOURIER_EDGES, gray, type Site } from '../lib/cove';
import { SIZE, spatialBand } from '../lib/features';
import { rackSites, rackState } from '../lib/cove-visualization';
import { CORTICAL_BANDS } from '../lib/cortical';
import { SiteOverlay } from './cove-science';

const fmt=(n:number)=>Number(n.toPrecision(4)).toString();

const rackColors = ['#67ddff','#a9eb72','#ffc170','#fb90d6','#9eaaff','#55efbd','#ff947e','#f7df72',
  '#7dbfff','#d5a5ff','#76ebdd','#ffa799','#b4df8a','#f7a7ea','#82c9ff','#ffe591'];

export const SignalRack = memo(function SignalRack({field,sites,site,mode,cursor,rate,seek,select,setMode,count,expanded=false,onExpand,cortical=false,children}:{
  field:Float64Array;sites:Site[];site:number;mode:number;cursor:number;rate:number;count:number;
  seek:(sample:number)=>void;select:(site:number)=>void;setMode:(mode:number)=>void;
  expanded?:boolean;onExpand?:()=>void;cortical?:boolean;children?:React.ReactNode;
}) {
  const bands=cortical?CORTICAL_BANDS:MODES;
  const [zoom,setZoom]=useState(1);
  const [quantity,setQuantity]=useState<'real'|'imaginary'|'amplitude'|'phase'>('real');
  const [hover,setHover]=useState<{row:number;sample:number}>();
  const channels=useMemo(()=>rackSites(Math.sqrt(sites.length),site,expanded)
    .flatMap((i, group)=>MODES.map((_,m)=>({site:i,mode:m,color:rackColors[(group*5+m)%rackColors.length]}))),
    [sites.length,site,expanded]);
  const span=Math.min(count,Math.max(3,Math.ceil(Math.min(count,Math.ceil(rate*.5))/zoom)));
  const from=Math.max(-1,Math.min(count-span-1,cursor-Math.floor(span/2)));
  const end=from+span;
  const dy=expanded?24:13.2, top=5, height=top+channels.length*dy+28, left=expanded?206:70, width=expanded?1120:680;
  const right=width-22;
  const x=(i:number)=>left+(i-from)/span*(right-left);
  const read=(r:{site:number;mode:number}, time:number)=>{
    const index=(Math.round(time)*sites.length*MODES.length+r.site*MODES.length+r.mode)*2;
    const {re,im}=cortical?{re:time<0?0:field[index],im:time<0?0:field[index+1]}:
      rackState(field,sites.length,r.site,r.mode,rate,time);
    const amplitude=Math.hypot(re,im);
    return quantity==='real'?re:quantity==='imaginary'?im:quantity==='amplitude'?amplitude:
      amplitude>1e-9?Math.atan2(im,re):null;
  };
  const paths=useMemo(()=>channels.map((r,i)=>{
    const subdivisions=cortical?1:8;
    const values=Array.from({length:span*subdivisions+1},(_,k)=>read(r,from+k/subdivisions));
    let min=Infinity,max=-Infinity;
    for(const v of values) if(v!==null){min=Math.min(min,v);max=Math.max(max,v);}
    if(min===Infinity){min=0;max=0;}
    if(quantity==='phase'){min=-Math.PI;max=Math.PI;}
    const center=(max+min)/2, half=Math.max(1e-9,(max-min)/2), baseline=top+(i+.5)*dy;
    let last:number|null=null;
    const d=values.map((v,k)=>{
      if(v===null){last=null;return '';}
      const move=last===null||(quantity==='phase'&&Math.abs(v-last)>Math.PI);
      last=v;
      return `${move?'M':'L'}${x(from+k/subdivisions).toFixed(2)} ${(baseline-(dy/2-1.25)*(v-center)/half).toFixed(2)}`;
    }).join('');
    return {d,min,max};
  // The horizontal transform and state accessor depend on these inputs.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }),[channels,field,sites.length,rate,quantity,from,span,dy,left,right,cortical]);
  const point=(event:React.PointerEvent<SVGSVGElement>|React.MouseEvent<SVGSVGElement>)=>{
    const rect=event.currentTarget.getBoundingClientRect();
    const col=Math.round(from+((event.clientX-rect.left)/rect.width*width-left)/(right-left)*span);
    const row=Math.floor(((event.clientY-rect.top)/rect.height*height-top)/dy);
    return {row:Math.max(0,Math.min(channels.length-1,row)),sample:Math.max(-1,Math.min(end,col))};
  };
  const active=hover?channels[hover.row]:undefined;
  const v=active&&hover?read(active,hover.sample):null;
  const unit=quantity==='phase'?'rad':'a.u.';
  return <section className={`signal-instrument ${expanded?'rack-expanded':'rack-primary'}`} aria-label={expanded?'Expanded COVE waveform rack':'COVE waveform rack'}>
    <div className="instrument-heading"><div><h2>{expanded?'Expanded waveform rack':'COVE waveform rack'}</h2>
      <p>{channels.length} traces · {channels.length/MODES.length} sites · {fmt(rate)} integrated samples/s</p></div>
      {children}
      {onExpand&&<button type="button" aria-expanded={expanded} onClick={onExpand}>{expanded?'Close expanded view':'Expand to 64 traces'}</button>}
      {expanded&&<div className="science-controls">
        <label>Signal quantity<select value={quantity} onChange={e=>setQuantity(e.target.value as typeof quantity)}>
          <option value="real">{cortical?'Excitatory deviation':'Real component'}</option><option value="imaginary">{cortical?'Inhibitory deviation':'Imaginary component'}</option>
          <option value="amplitude">{cortical?'Population-plane magnitude':'Magnitude envelope'}</option><option value="phase">{cortical?'Population-plane angle':'Wrapped phase'}</option>
        </select></label>
        <label>Time zoom · {zoom}×<input type="range" min="1" max="16" value={zoom} onChange={e=>setZoom(e.target.valueAsNumber)}/></label>
      </div>}
    </div>
    <figure className="cove-figure rack-figure"><figcaption>{cortical ? ({real:'Excitatory population deviation · simulated',imaginary:'Inhibitory population deviation',amplitude:'Population-plane magnitude',phase:'Population-plane angle'}[quantity]) : ({real:'Real component of rotating COVE states',imaginary:'Imaginary component',amplitude:'Magnitude envelope',phase:'Wrapped phase'}[quantity])}
      <span>Independent row scales · {unit}</span></figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{height}} role="img" aria-label={`${channels.length} stacked COVE signal traces`}
        onPointerMove={e=>setHover(point(e))} onPointerLeave={()=>setHover(undefined)} onClick={e=>{
          const p=point(e);select(channels[p.row].site);setMode(channels[p.row].mode);seek(Math.max(0,p.sample));
        }}>
        {channels.map((r,i)=><g key={`${r.site}:${r.mode}`}>
          <rect x="0" y={top+i*dy} width={width} height={dy} fill={r.site===site?(r.mode===mode?'#354f67':'#20394f'):Math.floor(i/4)%2?'#14263b':'#102034'}/>
          {i%4===0&&<path d={`M0 ${top+i*dy}H${right}`} stroke="#354b62"/>}
          <path d={`M${left} ${top+(i+.5)*dy}H${right}`} stroke="#2b4058" strokeDasharray="2 4"/>
          <text x="8" y={top+i*dy+dy/2+3} style={{fill:r.color}}>S{r.site+1} · {bands[r.mode]} Hz</text>
          {expanded&&<text x="78" y={top+i*dy+dy/2+3} className="rack-scale-text">{fmt(paths[i].min)}…{fmt(paths[i].max)}</text>}
        </g>)}
        {[0,1,2,3,4].map(j=>{
          const t=from+(end-from)*j/4;
          return <g key={j}><path d={`M${x(t)} ${top}V${height-25}`} stroke="#3b5169" opacity=".65"/>
            <text x={x(t)} y={height-9} className="rack-scale-text" textAnchor={j===4?'end':'middle'}>{fmt((t+1)/rate*1000)} ms</text></g>;
        })}
        {channels.map((r,i)=><path key={`${r.site}:${r.mode}`} d={paths[i].d} className="trace" style={{color:r.color}}/>)}
        {from===-1&&<line x1={x(-1)} x2={x(-1)} y1={top} y2={height-27} stroke="#b1def6" strokeWidth="2"/>}
        <line x1={x(cursor)} x2={x(cursor)} y1={top} y2={height-27} stroke="#fff0aa" opacity=".85"/>
        {hover&&<line x1={x(hover.sample)} x2={x(hover.sample)} y1={top} y2={height-27} stroke="#e4f4ff" strokeDasharray="3 3"/>}
      </svg>
      <p className="rack-readout">{active&&hover ?
        `Site ${active.site+1} · ${bands[active.mode]} Hz · ${fmt((hover.sample+1)/rate*1000)} ms · ${v===null?'undefined':fmt(v)} ${unit}` :
        '0 ms marks stimulus onset. Hover for exact samples. Click a row to select its site, channel and time.'}</p>
    </figure>
    {expanded&&<p>{cortical ? 'The frequency labels are nominal resonance settings. Angle describes the E–I state plane. Each row uses its own vertical scale.' : 'Substep curves interpolate the rotating complex state between integrated samples. Each row is centered and scaled to its displayed range. Hover values retain the original units.'}</p>}
  </section>;
});

export function AlignedMap({title,unit,sites,values,site,select,phase=false}:{
  title:string;unit:string;sites:Site[];values:(number|null)[];site:number;select:(i:number)=>void;phase?:boolean;
}) {
  const peak=phase?Math.PI:Math.max(1e-9,...values.map(v=>Math.abs(v??0)));
  return <figure className="cove-figure aligned-field"><figcaption>{title}<span>{unit}</span></figcaption>
    <div className="spatial-square">
      <svg viewBox="0 0 64 64" role="img" aria-label={title}>{sites.map((p,i)=>{
        const v=values[i], a=(v??0)/peak;
        return <circle key={i} cx={p.x+.5} cy={p.y+.5} r={sites.length>100?1.05:sites.length>25?1.4:2}
          fill={v===null?'#7f8c98':phase?`hsl(${240+180*a} 72% 49%)`:`hsl(${209-38*a} 76% ${94-62*a}%)`}
          stroke={i===site?'#d08a39':'#38546c'} strokeWidth={i===site?.68:.16}>
          <title>{`Site ${i+1} · ${v===null?'undefined':fmt(v)} ${unit}`}</title></circle>;
      })}</svg>
      <SiteOverlay geometry={sites} site={site} select={select}/>
    </div>
    <div className={phase?'science-scale phase-scale':'science-scale'}><span>{phase?'−π':'0'}</span><i/><span>{phase?'+π':fmt(peak)} {unit}</span></div>
    <p className="rack-readout">Site {site+1} · {values[site]===null?'undefined':fmt(values[site]!)} {unit} · computed samples</p>
  </figure>;
}

export function FeatureAtlas({rgb,sites,site,select,brightness,contrast,detail}:{
  rgb:Uint8Array;sites:Site[];site:number;select:(i:number)=>void;
  brightness:number[];contrast:number[];detail:number[];
}) {
  const [orientation,setOrientation]=useState(0);
  const [scale,setScale]=useState(0);
  const [band,setBand]=useState(1);
  const edges=[0,...FOURIER_EDGES,Math.SQRT2];
  const filtered=useMemo(()=>spatialBand(gray(rgb),edges[band],edges[band+1]),[rgb,band]);
  const peak=Math.max(1e-9,...filtered.map(Math.abs));
  const paths=Array.from({length:32},():string[]=>[]);
  filtered.forEach((v,i)=>{
    const level=Math.min(31,Math.max(0,Math.round((v/peak+1)*15.5)));
    paths[level].push(`M${i%SIZE} ${Math.floor(i/SIZE)}h1v1h-1z`);
  });
  const response=sites.map((_,i)=>{
    const j=i*16+orientation*4+scale*2;
    return Math.hypot(detail[j],detail[j+1]);
  });
  return <section className="science-stack feature-atlas"><h2>Image aligned feature atlas</h2>
    <p>Maps retain the input coordinate system. The Fourier image is a signed projection of the processed image. Orientation, contrast and luminance show the sampled features used by the encoder.</p>
    <div className="science-controls">
      <label>Fourier band<select value={band} onChange={e=>setBand(Number(e.target.value))}>
        {edges.slice(0,-1).map((low,i)=><option key={low} value={i}>{fmt(low)} to {fmt(Math.min(Math.SQRT1_2,edges[i+1]))} cycles/pixel</option>)}</select></label>
      <label>Filter orientation<select value={orientation} onChange={e=>setOrientation(Number(e.target.value))}>
        {[0,45,90,135].map((angle,i)=><option key={angle} value={i}>{angle}°</option>)}</select></label>
      <label>Orientation scale<select value={scale} onChange={e=>setScale(Number(e.target.value))}>
        <option value={0}>Lower spatial frequency</option><option value={1}>Higher spatial frequency</option>
      </select></label>
    </div>
    <div className="science-columns">
      <figure className="cove-figure aligned-field"><figcaption>Spatial frequency decomposition<span>Signed Fourier band · luminance</span></figcaption>
        <div className="spatial-square"><svg viewBox="0 0 64 64" role="img" aria-label="Image aligned inverse Fourier band">
          {paths.map((p,i)=><path key={i} d={p.join('')} fill={i<16?`hsl(26 65% ${40+i*3.4}%)`:`hsl(208 70% ${94-(i-16)*3.7}%)`}/>)}
        </svg><SiteOverlay geometry={sites} site={site} select={select}/></div>
        <div className="science-scale signed-scale"><span>{fmt(-peak)}</span><i/><span>{fmt(peak)} luminance</span></div>
      </figure>
      <AlignedMap title="Orientation response" unit="a.u." values={response} sites={sites} site={site} select={select}/>
      <AlignedMap title="Local contrast" unit="luminance" values={contrast} sites={sites} site={site} select={select}/>
      <AlignedMap title="Local luminance" unit="luminance" values={brightness} sites={sites} site={site} select={select}/>
    </div>
  </section>;
}

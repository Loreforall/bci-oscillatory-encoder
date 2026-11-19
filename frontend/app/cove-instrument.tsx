'use client';

import { memo, useMemo, useState } from 'react';
import { MODES, FOURIER_EDGES, gray, type Site } from '../lib/cove';
import { SIZE, spatialBand } from '../lib/features';
import { fieldIndex, MODE_COLORS } from '../lib/cove-visualization';
import { SiteOverlay } from './cove-science';

const fmt=(n:number)=>Number(n.toPrecision(4)).toString();

export const SignalRack = memo(function SignalRack({field,sites,site,mode,cursor,rate,seek,select,count}:{
  field:Float64Array;sites:Site[];site:number;mode:number;cursor:number;rate:number;count:number;
  seek:(sample:number)=>void;select:(site:number)=>void;
}) {
  const [zoom,setZoom]=useState(1);
  const [quantity,setQuantity]=useState<'real'|'imaginary'|'amplitude'|'phase'>('real');
  const [hover,setHover]=useState<{row:number;sample:number}>();
  const [arrangement,setArrangement]=useState<'montage'|'field'|'modes'>('montage');
  const channels=useMemo(()=>{
    const side=Math.sqrt(sites.length);
    const across=[0,.5,1].map(p=>Math.round(p*(side-1)));
    const anchors=across.flatMap(y=>across.map(x=>y*side+x));
    const montageSites=[...new Set([site,anchors[0],anchors[2],anchors[6],anchors[8]])].slice(0,4);
    const pairs=arrangement==='montage' ? montageSites.flatMap(i=>MODES.map((_,m)=>({site:i,mode:m}))) :
      arrangement==='field' ? [...new Set([site,...anchors])].map(i=>({site:i,mode})) :
      MODES.map((_,i)=>({site,mode:i}));
    return pairs.map(pair=>({...pair,data:Array.from({length:count},(_,t)=>{
      const at=fieldIndex(t,pair.site,pair.mode,sites.length), re=field[at], im=field[at+1], a=Math.hypot(re,im);
      return quantity==='real'?re:quantity==='imaginary'?im:quantity==='amplitude'?a:a>1e-9?Math.atan2(im,re):null;
    })}));
  },[field,sites.length,site,mode,quantity,count,arrangement]);
  const shown=Math.min(count,Math.max(3,Math.ceil(count/zoom)));
  const from=Math.max(0,Math.min(count-shown,cursor-Math.floor(shown/2)));
  const end=from+shown-1;
  const dy=arrangement==='montage'?38:46, top=27, height=top+channels.length*dy+34, left=120, right=1180;
  const x=(i:number)=>left+(i-from)/Math.max(1,end-from)*(right-left);
  const paths=useMemo(()=>channels.map((r,i)=>{
    const peak=quantity==='phase'?Math.PI:Math.max(1e-9,...r.data.slice(from,end+1).map(v=>Math.abs(v??0)));
    const mid=top+(i+.5)*dy;
    let last:number|null=null;
    let d='';
    for(let t=from;t<=end;t++){
      const v=r.data[t];
      if(v===null){last=null;continue;}
      const move=last===null||(quantity==='phase'&&Math.abs(v-last)>Math.PI);
      d+=`${move?'M':'L'}${(left+(t-from)/Math.max(1,end-from)*(right-left)).toFixed(2)} ${(mid-(dy/2-4)*v/peak).toFixed(2)}`;
      last=v;
    }
    return {d,peak};
  }),[channels,quantity,from,end,dy]);
  const point=(event:React.PointerEvent<SVGSVGElement>|React.MouseEvent<SVGSVGElement>)=>{
    const rect=event.currentTarget.getBoundingClientRect();
    const col=Math.round(from+((event.clientX-rect.left)/rect.width*1200-left)/(right-left)*(end-from));
    const row=Math.floor(((event.clientY-rect.top)/rect.height*height-top)/dy);
    return {row:Math.max(0,Math.min(channels.length-1,row)),sample:Math.max(from,Math.min(end,col))};
  };
  const active=hover?channels[hover.row]:undefined;
  return <section className="signal-instrument" aria-labelledby="signal-instrument-title">
    <div className="instrument-heading"><div><h2 id="signal-instrument-title">Distributed oscillatory signal rack</h2>
      <p>{sites.length} computed sites · {fmt(rate)} integration samples/s · {fmt(count/rate)} s modeled response</p></div>
      <div className="science-controls">
        <label>Channels<select value={arrangement} onChange={e=>setArrangement(e.target.value as typeof arrangement)}>
          <option value="montage">Four modes across four sites</option><option value="field">Compare scene sites</option><option value="modes">Compare modes at selected site</option>
        </select></label>
        <label>Signal quantity<select value={quantity} onChange={e=>setQuantity(e.target.value as typeof quantity)}>
          <option value="real">Real component</option><option value="imaginary">Imaginary component</option>
          <option value="amplitude">Magnitude envelope</option><option value="phase">Wrapped phase</option>
        </select></label>
        <label>Signal zoom · {zoom}×<input type="range" min="1" max="16" value={zoom} onChange={e=>setZoom(e.target.valueAsNumber)}/></label>
      </div></div>
    <figure className="cove-figure rack-figure"><figcaption>{arrangement==='montage'?'Distributed four-mode montage':arrangement==='field'?`${MODES[mode]} Hz across the scene`:`Four modes at site ${site+1}`}
      <span>{quantity==='phase'?'Radian scale':'Independent per channel scales · arbitrary units'}</span></figcaption>
      <svg viewBox={`0 0 1200 ${height}`} preserveAspectRatio="none" style={{height}} role="img" aria-label="Full resolution stacked COVE signal traces"
        onPointerMove={e=>setHover(point(e))} onPointerLeave={()=>setHover(undefined)} onClick={e=>{
          const p=point(e);select(channels[p.row].site);seek(p.sample);
        }}>
        {channels.map((r,i)=><g key={`${r.site}:${r.mode}`}>
          <rect x="0" y={top+i*dy} width="1200" height={dy} fill={r.site===site?'#e4f2fa':i%2?'#f1f7fa':'#f9fcfe'}/>
          <path d={`M${left} ${top+(i+.5)*dy}H${right}`} stroke="#b8ccd9" strokeDasharray="4 4"/>
          <text x="10" y={top+i*dy+17}>Site {r.site+1} · {MODES[r.mode]} Hz</text>
          <text x="10" y={top+i*dy+32}>±{fmt(paths[i].peak)} {quantity==='phase'?'rad':'a.u.'}</text>
        </g>)}
        {[0,1,2,3,4,5].map(j=>{
          const t=from+(end-from)*j/5;
          return <g key={j}><path d={`M${x(t)} ${top}V${height-32}`} stroke="#bcd1df" opacity=".55"/>
            <text x={x(t)} y={height-12} textAnchor="middle">{fmt((t+1)/rate)} s</text></g>;
        })}
        {channels.map((r,i)=><path key={`${r.site}:${r.mode}`} d={paths[i].d} className="trace" style={{color:MODE_COLORS[r.mode]}}/>)}
        <line x1={x(cursor)} x2={x(cursor)} y1={top} y2={height-34} stroke="#a65d21" strokeWidth="1.5"/>
        {hover&&<line x1={x(hover.sample)} x2={x(hover.sample)} y1={top} y2={height-34} stroke="#143c60" strokeDasharray="3 3"/>}
      </svg>
      <p className="rack-readout">{active&&hover ?
        `Site ${active.site+1} · ${MODES[active.mode]} Hz · ${fmt((hover.sample+1)/rate)} s · ${active.data[hover.sample]===null?'undefined':fmt(active.data[hover.sample]!)}` :
        'Hover to inspect a sample. Select a trace to set the site and playback time.'}</p>
    </figure>
    <p>These traces are the integrated complex state in a model that already includes mode rotation. The real component is displayed directly. Magnitude and wrapped phase describe the same state in polar coordinates.</p>
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

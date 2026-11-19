'use client';

import { memo, useMemo, useState } from 'react';
import { analyzeCove, crossMode, MODES, type Site, samplingTaps, spectrogram, spectrumWindowLength } from '../lib/cove';
import { fieldIndex, MODE_COLORS, modeTrajectory, phaseDifference, polar, spatialMatrix } from '../lib/cove-visualization';

const fmt = (n: number) => Number(n.toPrecision(4)).toString();
type Row = ReturnType<typeof analyzeCove>['rows'][number];
type Series = {name: string; color: string; data: (number|null)[]; dashed?: boolean};
type Window = [number, number];

export const ScientificPlot = memo(function ScientificPlot({title, unit, series, rate, domain, cursor, onSeek,
  range, offset = 1, xUnit = 's', wrapped = false, scaling = 'shared'}: {
  title: string; unit: string; series: Series[]; rate: number; domain: Window; cursor?: number;
  onSeek?: (sample:number) => void; range?: Window; offset?: number; xUnit?: string;
  wrapped?: boolean; scaling?: 'shared'|'independent'|'normalized';
}) {
  const [hover, setHover] = useState<number>();
  const [from,to] = domain;
  const low = range?.[0], high = range?.[1];
  const x = (n:number) => 64+(n-from)/Math.max(1,to-from)*674;
  const paths = useMemo(() => {
    const maxima = series.map(line => line.data.slice(from,to+1).reduce<number>((p,v)=>Math.max(p,Math.abs(v ?? 0)),1e-9));
    const maximum = Math.max(1e-9,...maxima);
    return series.map((line,i) => {
      const scale = scaling === 'shared' ? maximum : maxima[i];
      const min = scaling === 'normalized' ? -1 : low ?? -scale, max = scaling === 'normalized' ? 1 : high ?? scale;
      let previous: number|null = null;
      const commands: string[] = [];
      for (let n=from;n<=to;n++) {
        const raw = line.data[n];
        if (raw == null) {previous=null; continue;}
        const y = 136-122*((scaling==='normalized' ? raw/scale : raw)-min)/Math.max(1e-10,max-min);
        commands.push(`${previous === null || (wrapped && Math.abs(raw-previous)>Math.PI) ? 'M':'L'}${(64+(n-from)/Math.max(1,to-from)*674).toFixed(2)} ${y.toFixed(2)}`);
        previous = raw;
      }
      return {d:commands.join(' '), min,max, name:line.name};
    });
  },[series,from,to,low,high,wrapped,scaling]);
  const selected = hover ?? cursor;
  const hoverIndex = (clientX:number, element:SVGSVGElement) => {
    const bounds = element.getBoundingClientRect();
    const p = ((clientX-bounds.left)/bounds.width*752-64)/674;
    return Math.min(to,Math.max(from,Math.round(from+p*(to-from))));
  };
  return <figure className="cove-figure scientific-trace">
    <figcaption>{title}<span>{unit}{scaling === 'normalized' ? ' · divided by visible peak' : scaling === 'independent' ? ' · individual scales' : ''}</span></figcaption>
    <svg viewBox="0 0 752 164" preserveAspectRatio="none" role="img" aria-label={title}
      onPointerMove={e=>setHover(hoverIndex(e.clientX,e.currentTarget))}
      onPointerLeave={()=>setHover(undefined)}
      onClick={onSeek ? e=>onSeek(hoverIndex(e.clientX,e.currentTarget)) : undefined}>
      {[0,1,2,3,4].map(i => {
        const tx = 64+674*i/4, ty = 14+122*i/4;
        return <g key={i}><path className="grid" d={`M64 ${ty}H738 M${tx} 14V136`}/>
          <text x={tx} y="155" textAnchor="middle">{fmt((from+(to-from)*i/4+offset)/rate)} {xUnit}</text>
          {scaling !== 'independent' && <text x="59" y={ty+4} textAnchor="end">{fmt(paths[0].max-(paths[0].max-paths[0].min)*i/4)}</text>}
        </g>;
      })}
      {paths.map((path,i)=><path key={path.name} className="trace" style={{color:series[i].color}}
        strokeDasharray={series[i].dashed ? '4 3' : undefined} d={path.d}/>)}
      {cursor !== undefined && cursor>=from && cursor<=to && <line className="cove-cursor" x1={x(cursor)} x2={x(cursor)} y1="14" y2="136"/>}
      {hover !== undefined && <line stroke="#23425f" strokeDasharray="2 3" x1={x(hover)} x2={x(hover)} y1="14" y2="136"/>}
    </svg>
    <div className="science-legend" aria-label="Numerical trace readout">
      {selected !== undefined && <span>{fmt((selected+offset)/rate)} {xUnit}</span>}
      {series.map((line,i) => <span key={line.name} style={{color:line.color}}>{line.name}
        {selected !== undefined && ` · ${line.data[selected] == null ? 'undefined' : fmt(line.data[selected]!)}`}
        {scaling === 'independent' && ` · axis ±${fmt(paths[i].max)}`}</span>)}
    </div>
  </figure>;
});

export function SiteOverlay({geometry: sites, site, select, taps = false}: {geometry:Site[];site:number;select:(site:number)=>void;taps?:boolean}) {
  return <svg className="site-overlay" viewBox="0 0 64 64" preserveAspectRatio="none"
    aria-label="Retinotopic samples on visual input">
    {taps && samplingTaps(site,Math.sqrt(sites.length)).map(({x,y,weight})=><rect key={`${x},${y}`} x={x} y={y} width="1" height="1"
      fill="#69f2da" opacity={.25+.75*weight}><title>{`Sampling pixel (${x}, ${y}) · bilinear weight ${fmt(weight)}`}</title></rect>)}
    {sites.map((point,i)=><g key={i} role="button" tabIndex={0} aria-label={`Select visual site ${i+1}`}
      aria-pressed={i===site} onClick={()=>select(i)} onKeyDown={event=>{
        if (event.key==='Enter' || event.key===' ') {event.preventDefault();select(i);}
      }}><circle cx={point.x+.5} cy={point.y+.5} r={sites.length>100?1.05:2.35} fill="transparent"/>
      <circle className="site-spot" cx={point.x+.5} cy={point.y+.5} r={i===site?1.3:sites.length>100?.38:.73}
        fill={i===site?'#ffe397':'#e1f9ff'} stroke="#142e46" strokeWidth=".45"><title>{`Site ${i+1} · x ${fmt(point.x)}, y ${fmt(point.y)} pixels`}</title></circle></g>)}
  </svg>;
}

const phaseColor = (phase:number|null) => phase===null ? '#737f89' : `hsl(${(phase+Math.PI)/Math.PI*180+210} 66% 48%)`;

function SpatialMap({geometry: sites,states,site,select,peak,phase=false}: {
  geometry:Site[];
  states: {amplitude:number;phase:number|null;real:number;imaginary:number}[];
  site:number;select:(i:number)=>void;peak:number;phase?:boolean;
}) {
  return <figure className="cove-figure spatial-science"><figcaption>{phase?'Spatial phase':'Spatial amplitude'}
    <span>{phase?'Computed sites only':'Computed amplitudes'}</span></figcaption>
    <div className="spatial-square">
      <svg viewBox="0 0 64 64" role="img" aria-label={phase?'Phase at sampled sites':'Amplitude at sampled sites'}>
        {sites.map((p,i)=>{
          const level=states[i].amplitude/peak;
          return <circle key={i} cx={p.x+.5} cy={p.y+.5} r={site===i?1.55:sites.length>100?.87:1.2}
            fill={phase?phaseColor(states[i].phase):`hsl(${204-32*level} 70% ${94-58*level}%)`}
            stroke={site===i?'#a65f18':'#214965'} strokeWidth={site===i?.75:.24}/>;
        })}
      </svg>
      <SiteOverlay geometry={sites} site={site} select={select}/>
    </div>
    <div className={phase?'science-scale phase-scale':'science-scale'}>
      <span>{phase?'−π rad':'0 a.u.'}</span><i/><span>{phase?'+π rad':`${fmt(peak)} a.u.`}</span>
    </div>
    <p className="hint">Site {site+1} · {phase ? (states[site].phase===null?'undefined phase':`${fmt(states[site].phase!)} rad`) : `${fmt(states[site].amplitude)} a.u.`}</p>
  </figure>;
}

function Matrix({title, data, labels, selected, onSelect, unit}: {
  title:string;data:(number|null)[][];labels:string[];selected:number;onSelect?:(i:number)=>void;unit:string;
}) {
  const [hover,setHover]=useState<[number,number]>();
  const n=data.length, cell=250/n, maximum=Math.max(1e-9,...data.flat().map(x=>Math.abs(x??0)));
  const pair=hover ?? [selected,selected], value=data[pair[0]][pair[1]];
  const coordinates=(event:React.PointerEvent<SVGSVGElement>|React.MouseEvent<SVGSVGElement>):[number,number]|undefined=>{
    const bounds=event.currentTarget.getBoundingClientRect();
    const x=Math.floor(((event.clientX-bounds.left)/bounds.width*290-30)/cell);
    const y=Math.floor(((event.clientY-bounds.top)/bounds.height*287-24)/cell);
    return x>=0&&x<n&&y>=0&&y<n?[y,x]:undefined;
  };
  return <figure className="cove-figure science-matrix"><figcaption>{title}<span>{unit}</span></figcaption>
    <svg viewBox="0 0 290 287" preserveAspectRatio="none" role="img" aria-label={title}
      onPointerMove={event=>setHover(coordinates(event))} onPointerLeave={()=>setHover(undefined)}
      onClick={event=>{const point=coordinates(event);if(point)onSelect?.(point[0]);}}>
      <rect x="30" y="24" width="250" height="250" fill="#edf3f7"/>
      {data.flatMap((row,i)=>row.flatMap((v,j)=>v===null||v===0?[]:[<rect key={i*n+j}
        x={30+j*cell} y={24+i*cell} width={cell} height={cell}
        fill={v<0?`rgba(159,83,39,${.15+.85*Math.abs(v)/maximum})`:`rgba(20,105,153,${.13+.87*v/maximum})`}/>]))}
      <rect x="30" y={24+selected*cell} width="250" height={cell} fill="none" stroke="#c79548" strokeWidth=".6"/>
      {labels.map((label,i)=>(n===4 || i%Math.ceil(n/6)===0 || i===selected) && <g key={i}>
        <text x={30+(i+.5)*cell} y="18" textAnchor="middle">{label}</text>
        <text x="23" y={24+(i+.5)*cell+3} textAnchor="end">{label}</text>
      </g>)}
    </svg>
    <p className="hint">Columns are sources. Rows are targets. {labels[pair[1]]} → {labels[pair[0]]} · {value===null?'no direct link':fmt(value)}. Color maximum {fmt(maximum)}.</p>
  </figure>;
}

const SiteSpectrogram = memo(function SiteSpectrogram({power,rate,windowSize,seek}: {power:number[][];rate:number;windowSize:number;seek:(n:number)=>void}) {
  const stride=Math.max(1,Math.ceil(power.length/160)), bins=power[0].length;
  const hop=Math.max(1,Math.round(windowSize/2));
  const db=(v:number)=>10*Math.log10(Math.max(1e-12,v));
  const peak=Math.max(-120,...power.flat().map(db));
  const cells=Array.from({length:Math.ceil(power.length/stride)},(_,i)=>{
    const group=power.slice(i*stride,(i+1)*stride);
    return Array.from({length:bins},(_,j)=>db(group.reduce((sum,col)=>sum+col[j],0)/group.length));
  });
  return <figure className="cove-figure time-frequency"><figcaption>Selected mode spectrogram<span>Windowed power · dB</span></figcaption>
    <svg viewBox="0 0 460 206" role="img" aria-label="Mode spectrogram with time and frequency axes">
      {cells.flatMap((column,i)=>column.map((v,j)=>{
        const t=Math.max(0,(v-peak+55)/55);
        return <rect key={i*bins+j} x={44+400*i/cells.length} y={14+154*(bins-1-j)/bins}
          width={400/cells.length+.2} height={154/bins+.2} fill={`hsl(${225-42*t} 72% ${20+54*t}%)`}
          onClick={()=>seek(Math.round(i*stride*hop+(windowSize-1)/2))}>
          <title>{`${fmt((i*stride*hop+(windowSize+1)/2)/rate)} s · ${fmt(j*rate/windowSize)} Hz · ${fmt(v)} dB`}</title></rect>;
      }))}
      <text x="42" y="20" textAnchor="end">{fmt((bins-1)*rate/windowSize)} Hz</text>
      <text x="42" y="167" textAnchor="end">0 Hz</text>
      <text x="44" y="188">{fmt((windowSize+1)/2/rate)} s</text>
      <text x="444" y="188" textAnchor="end">{fmt(((power.length-1)*hop+(windowSize+1)/2)/rate)} s</text>
    </svg>
    <p className="hint">Hann windows of {windowSize} samples · hop {hop} samples · {fmt(rate/windowSize)} Hz bin spacing. Adjacent windows may be averaged for display.</p>
  </figure>;
});

const EvolutionCells = memo(function EvolutionCells({field, count, mode, rate, peak, seek, select, siteCount}: {
  field:Float64Array; count:number; mode:number; rate:number; peak:number; siteCount:number;
  seek:(i:number)=>void;select:(i:number)=>void;
}) {
  const hop=Math.max(1,Math.ceil(count/180)), columns=Math.ceil(count/hop);
  const groups=Array.from({length:40},():string[]=>[]);
  for(let x=0;x<columns;x++) {
    const start=x*hop, end=Math.min(count,start+hop);
    for(let i=0;i<siteCount;i++){
      let mean=0;
      for(let t=start;t<end;t++){
        const offset=fieldIndex(t,i,mode,siteCount);
        mean+=Math.hypot(field[offset],field[offset+1]);
      }
      mean/=(end-start);
      const level=Math.min(39,Math.round(39*mean/peak));
      groups[level].push(`M${(44+694*start/count).toFixed(2)} ${(14+i*120/siteCount).toFixed(2)}h${(694*(end-start)/count+.13).toFixed(2)}v${(120/siteCount+.06).toFixed(2)}h-${(694*(end-start)/count+.13).toFixed(2)}z`);
    }
  }
  return <g>{groups.map((paths,i)=><path key={i} d={paths.join('')} fill={`hsl(199 67% ${94-59*i/39}%)`}/>)}
    <rect x="44" y="14" width="694" height="120" fill="transparent" onClick={event=>{
      const b=event.currentTarget.getBoundingClientRect();
      const i=Math.min(siteCount-1,Math.max(0,Math.floor((event.clientY-b.top)/b.height*siteCount)));
      const t=Math.min(count-1,Math.max(0,Math.floor((event.clientX-b.left)/b.width*count)));
      select(i);seek(t);
    }}><title>{`Select a site and time across ${fmt(count/rate)} seconds`}</title></rect>
  </g>;
});

const ModeHeatmap=memo(function ModeHeatmap({values,count,rate,seek}: {values:number[][];count:number;rate:number;seek:(n:number)=>void}) {
  const max=Math.max(1e-9,...values.flat());
  const stride=Math.max(1,Math.ceil(count/240));
  return <g>{values.map((row,i)=><g key={i}>
    <text x="50" y={28+i*29} textAnchor="end" style={{fill:MODE_COLORS[i]}}>{MODES[i]} Hz</text>
    {Array.from({length:Math.ceil(count/stride)},(_,j)=>{
      const start=j*stride,end=Math.min(count,start+stride);
      const value=row.slice(start,end).reduce((a,b)=>a+b,0)/(end-start);
      return <rect key={j} x={56+686*start/count} y={10+i*29} width={686*(end-start)/count+.1} height="28"
        fill={`hsl(202 72% ${96-65*value/max}%)`} onClick={()=>seek(start)}>
        <title>{`${MODES[i]} Hz · ${fmt((start+1)/rate)} s · ${fmt(value)} a.u.`}</title></rect>;
    })}
  </g>)}<text x="56" y="143">{fmt(1/rate)} s</text><text x="400" y="143" textAnchor="middle">0 to {fmt(max)} a.u.</text>
  <text x="742" y="143" textAnchor="end">{fmt(count/rate)} s</text></g>;
});

export default function CoveScience({field, count, rate, sample, site, mode, row, rows, geometry: sites, select, setMode, seek}: {
  geometry:Site[];
  field:Float64Array;count:number;rate:number;sample:number;site:number;mode:number;row:Row;rows:Row[];
  select:(i:number)=>void;setMode:(i:number)=>void;seek:(i:number)=>void;
}) {
  const [quantity,setQuantity]=useState<'real'|'imaginary'|'amplitude'|'phase'>('real');
  const [scaling,setScaling]=useState<'shared'|'independent'|'normalized'>('shared');
  const [zoom,setZoom]=useState(1);
  const [reference,setReference]=useState(0);
  const [comparison,setComparison]=useState(false);
  const trajectories=useMemo(()=>MODES.map((_,i)=>modeTrajectory(field,site,i,sites.length)),[field,site,sites.length]);
  const modeData=trajectories[mode];
  const span=Math.max(2,Math.ceil(count/zoom));
  const start=Math.max(0,Math.min(count-span,sample-Math.floor(span/2)));
  const domain:Window=[start,Math.min(count-1,start+span-1)];
  const multichannel=useMemo(()=>trajectories.map((points,i)=>({name:`${MODES[i]} Hz`,color:MODE_COLORS[i],data:points.map(p=>p[quantity])})),[trajectories,quantity]);
  const componentSeries=useMemo(()=>[
    {name:'Real',color:MODE_COLORS[mode],data:modeData.map(v=>v.real)},
    {name:'Imaginary',color:'#65469a',data:modeData.map(v=>v.imaginary),dashed:true},
    {name:'Magnitude',color:'#087f79',data:modeData.map(v=>v.amplitude),dashed:true},
  ],[modeData,mode]);
  const phaseSeries=useMemo(()=>[{name:`${MODES[mode]} Hz relative to ${MODES[reference]} Hz`,color:MODE_COLORS[mode],
    data:modeData.map((p,i)=>phaseDifference(p.phase,trajectories[reference][i].phase))}],[modeData,mode,reference,trajectories]);
  const modalAmplitude=useMemo(()=>trajectories.map(points=>points.map(v=>v.amplitude)),[trajectories]);
  const matrix=useMemo(()=>spatialMatrix(row.weights),[row]);
  const lowIndex=fieldIndex(sample,site,0,sites.length);
  const links=MODES.map((_,i)=>MODES.map((__,j)=>i>0&&j===0 ? crossMode([field[lowIndex],field[lowIndex+1]],i).shiftHz:null));
  const spatial=(m:number)=>sites.map((_,i)=>{const idx=fieldIndex(sample,i,m,sites.length),real=field[idx],imaginary=field[idx+1];
    return {real,imaginary,...polar(real,imaginary)};
  });
  const peaks=useMemo(()=>MODES.map((_,i)=>{
    let peak=1e-8;
    for(let j=0;j<count;j++) for(let k=0;k<sites.length;k++){
      const offset=fieldIndex(j,k,i,sites.length);peak=Math.max(peak,Math.hypot(field[offset],field[offset+1]));
    }
    return peak;
  }),[field,count,sites.length]);
  const states=spatial(mode);
  const spectral=useMemo(()=>{
    const signal=modeData.map(v=>v.real);
    const power=spectrogram(signal,rate);
    const bins=power[0]?.length??0;
    const data=Array.from({length:bins},(_,i)=>10*Math.log10(Math.max(1e-12,power.reduce((sum,col)=>sum+col[i],0)/power.length)));
    return {power,data,window:spectrumWindowLength(signal.length,rate)};
  },[modeData,count,rate]);
  const temporal=useMemo(()=>[{name:'Motion proxy',color:'#176890',data:rows.map(r=>r.motion[site])},
    {name:'Local contrast',color:'#9a4d1f', data:rows.map(r=>r.contrast[site])}], [rows,site]);
  return <div className="science-stack">
    <section><h2>Oscillatory Dynamics</h2>
      <p>Every plotted point is a computed integration sample. The same playback cursor tracks all four modes. Click a time plot to seek or use the timeline. Hover to read exact samples.</p>
      <div className="science-controls">
        <label>Quantity<select value={quantity} onChange={e=>setQuantity(e.target.value as typeof quantity)}>
          <option value="real">Real component</option><option value="imaginary">Imaginary component</option>
          <option value="amplitude">Amplitude envelope</option><option value="phase">Wrapped phase</option>
        </select></label>
        <label>Vertical scaling<select value={scaling} disabled={quantity==='phase'} onChange={e=>setScaling(e.target.value as typeof scaling)}>
          <option value="shared">Shared scale</option><option value="independent">Independent scales</option><option value="normalized">Normalize visible peak</option>
        </select></label>
        <label>Time zoom · {zoom}×<input type="range" min="1" max="16" value={zoom} onChange={e=>setZoom(e.target.valueAsNumber)}/></label>
      </div>
      <ScientificPlot title="Four mode complex trajectory" unit={quantity==='phase'?'rad':'a.u.'} series={multichannel}
        rate={rate} domain={domain} cursor={sample} onSeek={seek} wrapped={quantity==='phase'}
        range={quantity==='phase'?[-Math.PI,Math.PI]:undefined} scaling={quantity==='phase'?'shared':scaling}/>
      <div className="science-mode-details" role="group" aria-label="Expand a mode">
        {MODES.map((hz,i)=><button key={hz} aria-pressed={i===mode} onClick={()=>setMode(i)}>{hz} Hz components</button>)}
      </div>
      <ScientificPlot title={`${MODES[mode]} Hz real, imaginary and amplitude`} unit="a.u." series={componentSeries}
        rate={rate} domain={domain} cursor={sample} onSeek={seek}/>
    </section>
    <section><h2>Spatial Field</h2>
      <p>Amplitude is the magnitude of the modeled complex state in arbitrary units. Each colored location shows a computed sample. Gray marks undefined phase.</p>
      <label className="science-toggle"><input type="checkbox" checked={comparison} onChange={e=>setComparison(e.target.checked)}/> Compare amplitude across four modes</label>
      <div className="science-columns">
        {(comparison?MODES.map((_,i)=>i):[mode]).map(i=><div key={i}><h3>{MODES[i]} Hz</h3>
          <SpatialMap geometry={sites} states={spatial(i)} site={site} select={j=>{select(j);setMode(i);}} peak={peaks[i]}/></div>)}
        <div><h3>{MODES[mode]} Hz</h3><SpatialMap geometry={sites} states={states} site={site} select={select} peak={peaks[mode]} phase/></div>
      </div>
      <figure className="cove-figure scientific-trace evolution-plot"><figcaption>Distributed amplitude through time
        <span>{MODES[mode]} Hz · rows are sites · 0 to {fmt(peaks[mode])} a.u.</span></figcaption>
        <svg viewBox="0 0 752 164" preserveAspectRatio="none" role="img" aria-label="Site by time amplitude heatmap">
          <EvolutionCells field={field} count={count} mode={mode} rate={rate} peak={peaks[mode]} seek={seek} select={select} siteCount={sites.length}/>
          <rect x="44" y={14+site*120/sites.length} width="694" height={120/sites.length} fill="none" stroke="#d08438" pointerEvents="none"/>
          <line className="cove-cursor" x1={44+694*(sample+.5)/count} x2={44+694*(sample+.5)/count} y1="14" y2="134" pointerEvents="none"/>
          {[0,.2,.4,.6,.8,1].map(p=>Math.floor(p*(sites.length-1))).map(i=><text key={i} x="39" y={18+i*120/sites.length} textAnchor="end">{i+1}</text>)}
          <text x="44" y="153">{fmt(1/rate)} s</text><text x="738" y="153" textAnchor="end">{fmt(count/rate)} s</text>
        </svg>
        <p className="hint">Select a cell to move to its site and time. For long clips each cell shows mean amplitude over neighboring samples.</p>
      </figure>
    </section>
    <section><h2>Coupling and Interactions</h2>
      <p>Spatial weights are model parameters for the current input frame. The mode matrix shows the computed frequency shift at the current sample, evaluated from the state for the next update. Empty entries indicate no direct link. These plots describe the model rather than measured neural connectivity.</p>
      <div className="science-columns"><Matrix title="Spatial coupling matrix" data={matrix} labels={sites.map((_,i)=>String(i+1))} selected={site} onSelect={select} unit="Unitless weight"/>
        <Matrix title="Cross frequency interaction matrix" data={links} labels={MODES.map(String)} selected={mode} onSelect={setMode} unit="Frequency shift · Hz"/></div>
      <label className="science-reference">Phase reference<select value={reference} onChange={e=>setReference(Number(e.target.value))}>
        {MODES.map((hz,i)=><option key={hz} value={i}>{hz} Hz</option>)}</select></label>
      <ScientificPlot title="Relative phase between modes" unit="rad" series={phaseSeries} rate={rate} domain={domain}
        cursor={sample} onSeek={seek} range={[-Math.PI,Math.PI]} wrapped/>
      <p>Relative phase is the target phase minus the reference phase, wrapped to ±π. Gaps indicate undefined phase. This quantity is distinct from physiological coherence.</p>
    </section>
    <section><h2>Signal Analysis</h2>
      <p>Frequency analysis uses the real component of the selected mode and site at {fmt(rate)} samples per second. The nominal mode label describes the model, rather than a measured EEG rhythm.</p>
      {spectral ? <div className="science-columns">
        <ScientificPlot title="Selected mode spectrum" unit="Mean window power · dB"
          series={[{name:`${MODES[mode]} Hz mode`,color:MODE_COLORS[mode],data:spectral.data}]}
          rate={spectral.window/rate} offset={0} xUnit="Hz" domain={[0,spectral.data.length-1]}
          range={[Math.min(...spectral.data)-2,Math.max(...spectral.data)+2]}/>
        {count/rate>=.5 ? <SiteSpectrogram power={spectral.power} rate={rate} windowSize={spectral.window} seek={seek}/> :
          <p className="science-note">The still image response spans {fmt(count/rate)} seconds. The spectrum describes that transient with {fmt(rate/spectral.window)} Hz bin spacing. A longer clip is needed for a time frequency sequence.</p>}
      </div> : <p className="science-note">This response spans {fmt(count/rate)} seconds. A spectrum or spectrogram would have limited frequency resolution, so the selected mode is shown as a transient trajectory above. Upload a longer clip for time frequency analysis.</p>}
      <figure className="cove-figure time-frequency"><figcaption>Mode amplitude versus time<span>Selected site · shared scale</span></figcaption>
        <svg viewBox="0 0 760 150" preserveAspectRatio="none" role="img" aria-label="Mode amplitude versus time">
          <ModeHeatmap values={modalAmplitude} count={count} rate={rate} seek={seek}/>
          <line x1={56+686*(sample+.5)/count} x2={56+686*(sample+.5)/count} y1="10" y2="126" className="cove-cursor" pointerEvents="none"/>
        </svg>
      </figure>
      {rows.length>1 && <ScientificPlot title="Visual features over time at selected site" unit="Normalized input features"
        series={temporal} rate={rate*rows.length/count} offset={0} domain={[0,rows.length-1]}
        cursor={Math.floor(sample*rows.length/count)}/>}
      <p className="hint">The complete integration interval is {fmt(count/rate)} seconds.</p>
    </section>
  </div>;
}

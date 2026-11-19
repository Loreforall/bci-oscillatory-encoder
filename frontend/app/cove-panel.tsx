'use client';

import { memo, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { analyzeCove, crossMode, flattenCode, GRID, HAAR_SCALES, MODES, makeSites, spectrogram, spectrumWindowLength } from '../lib/cove';
import { affine, type Calibration, ERP_BINS, parseCalibration, predictErp } from '../lib/cove-calibration';
import { imageError, refineCode } from '../lib/cove-reconstruction';
import { fieldCsv, fieldJson } from '../lib/cove-export';
import type { Input, Result } from '../lib/encoder';
import { playbackFrame, playbackSample } from '../lib/cove-playback';
import { AlignedMap, FeatureAtlas, SignalRack } from './cove-instrument';
import CoveScience, { SiteOverlay } from './cove-science';
import FramePreview from './frame-preview';
import renderer from '../lib/cove-decoder.json';

function InputPreview({rgb, fallback, label, overlay}: {rgb: Uint8Array; fallback?: boolean; label?: string; overlay?: ReactNode}) {
  return <figure className={fallback ? "cove-figure neighborhood-photo" : "cove-figure"}><figcaption>Visual input <span>{label ?? "Processed RGB"}</span></figcaption>
    {fallback ? <>
      {/* Local photograph and its encoder raster are derived from the same saved JPEG. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <div className="photo-overlay-wrap"><img src="/neighborhood.jpg" width={1280} height={853} alt="Daylight on a residential street with trees, houses and parked cars"/>{overlay}</div>
      <p className="photo-credit"><a href="https://commons.wikimedia.org/wiki/File:DFC_2054_A_quiet_narrow_neighborhood_street_lined_with_trees_and_parked_cars_sunlight_filtering_through_the_greenery.jpg">PattayaPatrol</a>
        {' · '}<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY SA 4.0</a> · resized for this demonstration</p>
    </> : overlay ? <div className="photo-overlay-wrap uploaded-overlay"><FramePreview rgb={rgb} label="Current visual input"/>{overlay}</div>
        : <div className="frame-view"><FramePreview rgb={rgb} label="Current visual input"/></div>}
  </figure>;
}

const DEFAULT_SLOPES = [2, 2];
function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const metric = (value: number) => Number(value.toPrecision(3)).toString();

const Heatmap = memo(function Heatmap({title, rows, min, max, unit, grayscale = false}: {
  title: string; rows: (number | null)[][]; min: number; max: number; unit: string; grayscale?: boolean;
}) {
  const height = rows.length, width = rows[0]?.length ?? 0;
  return <figure className="cove-figure">
    <figcaption>{title}<span>{unit}</span></figcaption>
    {height ? <svg viewBox="0 0 210 130" role="img" aria-label={title} preserveAspectRatio="none">
      {rows.flatMap((row, y) => row.map((value, x) => {
        const intensity = value === null ? 0 : Math.min(1, Math.max(0, (value-min)/(max-min || 1)));
        const fill = value === null ? '#30353c' : grayscale ? `hsl(0 0% ${100*intensity}%)` :
          `hsl(${195 - 155*intensity} 65% ${12+65*intensity}%)`;
        return <rect key={y*width+x} x={210*x/width} y={130*y/height} width={210/width+.1} height={130/height+.1}
          fill={fill}>
          <title>{`Row ${y+1}, column ${x+1}: ${value === null ? 'undefined phase' : value.toPrecision(4)}`}</title>
        </rect>;
      }))}
    </svg> : <p className="hint">Insufficient samples</p>}
  </figure>;
});

function Trace({title, values, unit, start, end, extent, cursor, color}: {
  title: string; values: number[]; unit: string; start: string; end: string; extent?: [number, number]; cursor?: number; color?: string;
}) {
  const loLimit = extent?.[0], hiLimit = extent?.[1];
  const {lo, hi, span, points} = useMemo(() => {
    const lo = loLimit ?? Math.min(0, ...values);
    const hi = hiLimit ?? Math.max(0, ...values);
    const span = Math.max(1e-12, hi-lo);
    return {lo, hi, span, points: values.map((v, i) =>
      `${29+i*171/Math.max(1,values.length-1)},${105-93*(v-lo)/span}`).join(' ')};
  }, [values, loLimit, hiLimit]);
  return <figure className="cove-figure"><figcaption>{title}<span>{unit}</span></figcaption>
    <svg viewBox="0 0 210 130" role="img" aria-label={title}>
      <path className="axes" d="M29 12V105H200"/>
      <polyline className="trace t0" style={{color}} points={points}/>
      {cursor !== undefined && <g><line className="cove-cursor" x1={29+171*cursor/Math.max(1,values.length-1)}
        x2={29+171*cursor/Math.max(1,values.length-1)} y1="12" y2="105"/>
        <circle cx={29+171*cursor/Math.max(1,values.length-1)} cy={105-93*(values[cursor]-lo)/span} r="3" fill="#8b4b1a"/></g>}
      <text x="27" y="16" textAnchor="end">{metric(hi)}</text>
      <text x="27" y="105" textAnchor="end">{metric(lo)}</text>
      <text x="29" y="124">{start}</text><text x="200" y="124" textAnchor="end">{end}</text>
    </svg>
  </figure>;
}

function grid(values: (number | null)[], width = GRID) {
  return Array.from({length: width}, (_, row) => values.slice(row * width, (row + 1) * width));
}

export default function CovePanel({frames, frame, source, onFrameChange, home, startAdvanced}: {
  frames: Uint8Array[]; frame: number; source: Input | Result["input"]; onFrameChange: (frame: number) => void;
  home?: {label: string; example: boolean; upload: ReactNode}; startAdvanced?: boolean;
}) {
  const [view, setView] = useState<'demo' | 'analysis'>(startAdvanced ? 'analysis' : 'demo');
  const [expanded, setExpanded] = useState(false);
  const [overlay, setOverlay] = useState(true);
  const [resetVersion, setResetVersion] = useState(0);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const clockStart = useRef(0);
  const isHome = Boolean(home);
  useEffect(() => {
    if (isHome && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) setPlaying(true);
  }, [isHome]);

  const [model, setModel] = useState<Calibration>();
  const [error, setError] = useState('');
  const [side, setSide] = useState(home ? 16 : GRID);
  const [site, setSite] = useState(home ? 119 : 12);
  const [mode, setMode] = useState(0);
  const [scale, setScale] = useState(0);
  const [electrode, setElectrode] = useState(0);
  const fileVersion = useRef(0);
  const stillImage = source.kind === "images" && frames.length === 1;
  const activeModel = stillImage && !home ? model : undefined;
  const slopes = activeModel?.coupling ?? DEFAULT_SLOPES;
  const analysis = useMemo(() => analyzeCove(frames, source.fps, slopes, true, side), [frames, source.fps, slopes, side]);
  const {rows, internal, internalRate, stepsPerFrame, field, sites} = analysis;
  const cursorMatchesFrame = playbackFrame(cursor, stepsPerFrame) === frame;
  const sample = cursorMatchesFrame ? cursor : frame * stepsPerFrame;

  useEffect(() => {
    if (!playing) return;
    if (!cursorMatchesFrame) {
      setPlaying(false);
      return;
    }
    let request = 0;
    let start: number | undefined;
    const from = clockStart.current;
    const tick = (now: number) => {
      start ??= now;
      const elapsed = now-start;
      const loopDuration = internal.length / internalRate * 1000;
      const next = home?.example
        ? playbackSample(0, (from/internalRate*1000+elapsed) % loopDuration, internalRate, internal.length)
        : playbackSample(from, elapsed, internalRate, internal.length);
      setCursor(next);
      onFrameChange(playbackFrame(next, stepsPerFrame));
      if (home?.example || next < internal.length-1) request = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    request = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(request);
  }, [playing, internalRate, internal.length, stepsPerFrame, onFrameChange, cursorMatchesFrame, home?.example]);

  const seek = useCallback((next: number) => {
    const index = Math.min(internal.length-1,Math.max(0,next));
    setPlaying(false);
    setCursor(index);
    onFrameChange(playbackFrame(index, stepsPerFrame));
  }, [internal.length, onFrameChange, stepsPerFrame]);

  const modeTrace = useMemo(() => Array.from({length: internal.length}, (_, t) =>
    field![(t*sites.length*MODES.length + site*MODES.length + mode)*2]),
    [field, internal.length, site, mode, sites.length]);
  const stateAt = (i: number, k: number): [number, number] => {
    const offset = (sample*sites.length*MODES.length + i*MODES.length + k)*2;
    return [field![offset], field![offset+1]];
  };
  const row = rows[frame];
  const windowStart = Math.floor(frame / 8) * 8;
  const referenceRows = useMemo(() => side === GRID ? rows :
    analyzeCove(frames, source.fps, slopes, true).rows, [frames,source.fps,slopes,side,rows]);
  const reconstruction = useMemo(() => {
    const window = referenceRows.slice(windowStart, windowStart + 8);
    const forward = window.map(r => flattenCode(r.state));
    const refined = refineCode(forward, window.map(r => r.pixels), window.map(r => r.weights), renderer);
    return window.map((r, i) => {
      const plain = affine(forward[i], renderer.weights, renderer.offset);
      const decoded = affine(refined[i], renderer.weights, renderer.offset);
      return {plain, decoded, initial: imageError(r.pixels, plain), final: imageError(r.pixels, decoded)};
    });
  }, [referenceRows, windowStart])[frame - windowStart];
  const {signal, spectrogramRows, spectrum, upperHz} = useMemo(() => {
    const signal = internal.map(sample => sample[site]);
    const power = spectrogram(signal, internalRate);
    const db = (value: number) => 10 * Math.log10(Math.max(1e-12, value));
    const bins = power[0]?.length ?? 0;
    const spectrum = Array.from({length: bins}, (_, k) => db(power.reduce((s, w) => s+w[k], 0) / power.length));
    const stride = Math.max(1, Math.ceil(power.length/240));
    const spectrogramRows = Array.from({length: bins}, (_, k) =>
      Array.from({length: Math.ceil(power.length/stride)}, (_, col) => {
        const windows = power.slice(col*stride, (col+1)*stride);
        return db(windows.reduce((sum, w) => sum+w[bins-1-k], 0)/windows.length);
      }));
    const upperHz = (bins - 1) * internalRate / Math.max(1, spectrumWindowLength(signal.length, internalRate));
    return {signal, spectrogramRows, spectrum, upperHz};
  }, [internal, internalRate, site]);
  const sliceSize = Math.min(modeTrace.length, Math.round(.6*internalRate));
  const sliceStart = Math.max(0, Math.min(modeTrace.length-sliceSize, sample-Math.floor(sliceSize/2)));
  const waveWindow = modeTrace;
  const waveCursor = sample;
  const wavePeak = useMemo(() => Math.max(1e-5, ...modeTrace.map(Math.abs)), [modeTrace]);
  const spectrumPeak = Math.max(-119, ...spectrogramRows.flat());
  const magnitude = sites.map((_, i) => Math.hypot(...stateAt(i, mode)));
  const phase = sites.map((_, i) => {
    const [re, im] = stateAt(i, mode);
    return magnitude[i] > 1e-9 ? Math.atan2(im, re) : null;
  });
  const {driveGain: gate, shiftHz} = crossMode(stateAt(site, 0), mode);
  const crossTrace = useMemo(() => {
    const gain: number[] = [], frequency: number[] = [];
    for (let t = 0; t < internal.length; t++) {
      const index = (t*sites.length + site) * MODES.length * 2;
      const coupling = crossMode([field![index], field![index+1]], mode);
      gain.push(coupling.driveGain);
      frequency.push(coupling.shiftHz);
    }
    return {gain, frequency};
  }, [field, internal.length, mode, site, sites.length]);
  const firstTime = `${metric(1/internalRate)} s`;
  const lastTime = `${metric(internal.length/internalRate)} s`;
  const prediction = useMemo(() => {
    if (!activeModel) return {erp: [], legacy: []};
    const code = activeModel.schema === 'cove-erp-v1'
      ? flattenCode(analyzeCove(frames, 24, activeModel.coupling).rows[0].state)
      : flattenCode(rows[0].state);
    const still = rows[0];
    return {
      erp: predictErp(activeModel, still.conventional,
        [...still.detail, ...still.spectrum.map(v => 4*v), ...still.pixels], code),
      legacy: activeModel.schema === 'cove-erp-v1'
        ? affine(code, activeModel.decoder, activeModel.decoder_offset) : [],
    };
  }, [activeModel, frames, rows]);
  const erp = prediction.erp.slice(electrode*ERP_BINS, (electrode+1)*ERP_BINS);
  const peak = Math.max(1e-9, ...erp.map(Math.abs));
  const waveform = <Trace title="Computed oscillatory waveform" values={waveWindow}
    unit={`Site ${site+1} · ${MODES[mode]} Hz · real component`}
    start={home ? `${metric((sliceStart+1)/internalRate)} s` : firstTime}
    end={home ? `${metric((sliceStart+sliceSize)/internalRate)} s` : lastTime}
    extent={home ? [-wavePeak,wavePeak] : undefined} cursor={waveCursor}/>;
  const activity = <figure className="cove-figure"><figcaption>Retinotopic activity<span>Selected mode amplitude</span></figcaption>
    <svg viewBox="0 0 64 64" aria-label="Amplitude at retinotopic sites" role="img">
      {sites.map((point, i) => <circle key={i} cx={point.x+.5} cy={point.y+.5}
        r={1.1+2.4*Math.sqrt(magnitude[i])} fill={i === site ? '#12638f' : '#529fad'}
        stroke={i === site ? '#203449' : '#96c4cc'} strokeWidth=".5">
        <title>{`Site ${i+1} · ${metric(magnitude[i])} a.u.`}</title>
      </circle>)}
    </svg>
  </figure>;
  return <section className={home ? "cove-panel cove-hero" : "cove-panel"} aria-label={home ? "Live COVE demonstration" : "Distributed COVE analysis"}>
    {!home && <>
    <h3>Distributed COVE <small>· simulated population activity</small></h3>
    <div className="cove-views" role="group" aria-label="COVE view">
      <button aria-pressed={view === 'demo'} onClick={() => setView('demo')}>Demonstration</button>
      <button aria-pressed={view === 'analysis'} onClick={() => setView('analysis')}>Advanced analysis</button>
    </div>
    </>}
    {home && <div className="cove-hero-grid">
      <div>
        <InputPreview rgb={frames[frame]} fallback={home.example} label={home.label}
          overlay={overlay && <SiteOverlay geometry={sites} site={site} select={setSite} taps/>}/>
        <div className="science-overlay-controls">
          <label className="science-toggle"><input type="checkbox" checked={overlay} onChange={event=>setOverlay(event.target.checked)}/> Show sampling sites</label>
          <span>Site {site+1} · x {metric(sites[site].x)}, y {metric(sites[site].y)} px on the 64 × 64 input</span>
        </div>
        <p className="hint">Highlighted pixels are the bilinear sampling taps for the selected location. Local image filters also use surrounding pixels. The markers indicate sampling locations.</p>
        <AlignedMap title={`${MODES[mode]} Hz whole scene amplitude`} unit="a.u." sites={sites} values={magnitude} site={site} select={setSite}/>
        <dl className="hero-site-readout">
          <dt>Local luminance</dt><dd>{metric(row.brightness[site])}</dd>
          <dt>Local contrast</dt><dd>{metric(row.contrast[site])}</dd>
          <dt>Signed motion proxy</dt><dd>{metric(row.motion[site])}</dd>
          <dt>Selected mode input magnitude</dt><dd>{metric(Math.hypot(...row.drive[site][mode]))}</dd>
          <dt>{MODES[mode]} Hz amplitude</dt><dd>{metric(magnitude[site])} a.u.</dd>
          <dt>{MODES[mode]} Hz phase</dt><dd>{phase[site]===null?'undefined':`${metric(phase[site]!)} rad`}</dd>
        </dl>
        <p className="demo-caption">{sites.length} independently computed sites. The higher resolution fields are computational extensions. Reported EEG results use the verified reference field.</p>
      </div>
      <SignalRack key={`primary-${resetVersion}`} field={field!} sites={sites} site={site} mode={mode} cursor={sample}
        rate={internalRate} count={internal.length} seek={seek} select={setSite} setMode={setMode} onExpand={()=>{
          setExpanded(true);
          requestAnimationFrame(()=>document.getElementById('expanded-waveforms')?.scrollIntoView({block:'start'}));
        }}/>
    </div>}
    <div className="cove-transport" aria-label="COVE playback controls">
      <button onClick={() => {
        if (playing) setPlaying(false);
        else {
          clockStart.current = sample >= internal.length-1 ? 0 : sample;
          setCursor(clockStart.current);
          onFrameChange(playbackFrame(clockStart.current, stepsPerFrame));
          setPlaying(true);
        }
      }}>{playing ? 'Pause' : 'Play'}</button>
      <button onClick={() => seek(0)}>Reset playback</button>
      {home && <button onClick={() => {
        seek(0); setSite(Math.floor((side-1)/2)*(side+1)); setMode(0); setScale(0); setExpanded(false); setOverlay(true);
        setResetVersion(value=>value+1);
      }}>Reset View</button>}
      <label>Playback timeline <input type="range" min="0" max={internal.length-1} value={sample}
        aria-valuetext={`${metric((sample+1)/internalRate)} seconds, input frame ${frame+1}`}
        onChange={event => seek(event.target.valueAsNumber)}/></label>
      <output aria-live="off">{metric((sample+1)/internalRate)} s · frame {frame+1}/{frames.length}</output>
    </div>
    <div className="cove-controls">
      {!home && "processed_frames_sha256" in source && <><button onClick={() => download(fieldJson(analysis, source, slopes, activeModel ? {
        subject: activeModel.subject, channels: activeModel.channels,
        onset_bin_centers_ms: Array.from({length: ERP_BINS}, (_, i) => 10+20*i),
        prediction_scaled_csd_flat: prediction.erp,
        held_out: activeModel.schema === 'cove-erp-v2' ? activeModel.held_out : null,
      } : null), 'cove-field.json')}>Export COVE JSON</button>
      <button onClick={() => download(fieldCsv(analysis, source), 'cove-frame-states.csv')}>Export COVE CSV</button></>}
      {home && <label>Computed field<select value={side} onChange={event=>{
        const next=Number(event.target.value);
        const previous=sites[site];
        let closest=0, distance=Infinity;
        makeSites(next).forEach((point,i)=>{
          const d=Math.hypot(point.x-previous.x,point.y-previous.y);
          if(d<distance){closest=i;distance=d;}
        });
        setSite(closest); setSide(next); seek(0);
      }}><option value={5}>5 × 5 reference</option><option value={10}>10 × 10 extension</option><option value={16}>16 × 16 extension</option></select></label>}
      <label>Site <select value={site} onChange={event => setSite(Number(event.target.value))}>
        {sites.map((_, index) => <option key={index} value={index}>{index+1}</option>)}
      </select></label>
      <label>Mode <select value={mode} onChange={event => setMode(Number(event.target.value))}>
        {MODES.map((hz, index) => <option key={hz} value={index}>{hz} Hz</option>)}
      </select></label>
      {(home || view === 'analysis') && <label>Haar scale <select value={scale} onChange={event => setScale(Number(event.target.value))}>
        {HAAR_SCALES.map((size, index) => <option key={size} value={index}>{size}+{size} frames</option>)}
      </select></label>}
      {!home && stillImage && <label>ERP calibration JSON <input type="file" accept=".json,application/json" onChange={async event => {
        const file = event.target.files?.[0];
        const version = ++fileVersion.current;
        setError('');
        setModel(undefined);
        if (!file) return;
        try {
          if (file.size > 8 * 1024**2) throw new Error('Calibration must be under 8 MiB.');
          const parsed = parseCalibration(await file.text());
          if (version === fileVersion.current) {
            setModel(parsed);
            setElectrode(0);
          }
        } catch (problem) {
          if (version === fileVersion.current) setError(problem instanceof Error ? problem.message : 'Calibration could not be loaded.');
        }
      }}/></label>}
      {activeModel && <label>Electrode <select value={electrode} onChange={event => setElectrode(Number(event.target.value))}>
        {activeModel.channels.map((label, index) => <option key={label} value={index}>{label}</option>)}
      </select></label>}
    </div>
    {error && <p className="error" role="alert">{error}</p>}
    {home && <>
      {expanded && <div id="expanded-waveforms">
        <SignalRack key={`rack-${resetVersion}`} field={field!} sites={sites} site={site} mode={mode} cursor={sample}
          rate={internalRate} count={internal.length} seek={seek} select={setSite} setMode={setMode} expanded onExpand={()=>{
            setExpanded(false);
            document.querySelector<HTMLButtonElement>('.rack-primary button')?.focus();
          }}/>
      </div>}
      {home.upload}
      <FeatureAtlas key={`atlas-${resetVersion}`} rgb={frames[frame]} sites={sites} site={site} select={setSite}
        brightness={row.brightness} contrast={row.contrast} detail={row.detail}/>
      <CoveScience key={resetVersion} field={field!} count={internal.length} rate={internalRate} sample={sample}
        site={site} mode={mode} row={row} rows={rows} geometry={sites} select={setSite} setMode={setMode} seek={seek}/>
      <section className="science-stack"><h2>Visual Feature Maps</h2>
        <p>Fourier bands summarize the entire image. Contrast, orientation drive and signed Haar detail are local feature maps on the modeled grid.</p>
        <div className="plot-grid">
          <Heatmap title="Fourier spatial bands" rows={[row.spectrum]} min={0} max={Math.max(.01,...row.spectrum)} unit="RMS · low to high"/>
          <Heatmap title="Local contrast" rows={grid(row.contrast,side)} min={0} max={.5} unit="Normalized luminance"/>
          <Heatmap title="Orientation drive magnitude" rows={grid(row.drive.map(modes=>Math.hypot(...modes[mode])),side)} min={0} max={1} unit="a.u."/>
          <Heatmap title="Temporal Haar detail" rows={grid(row.wavelets.map(v=>v[scale]),side)} min={-.25} max={.25}
            unit={frames.length===1?'Still input · zero detail':'Signed luminance change'}/>
          <Trace title="Mode drive multiplier over time" values={crossTrace.gain} unit="Unitless"
            start={firstTime} end={lastTime} cursor={sample} extent={[.82,1]}/>
        </div>
      </section>
      <section className="science-stack"><h2>Reconstruction</h2>
        <p>Both estimates use the verified 5 × 5 reference decoder. The forward estimate uses the reference state. Input guided refinement also uses the original pixels. The larger field has no trained decoder.</p>
        <div className="science-columns reconstruction-grid">
          <InputPreview rgb={frames[frame]}/>
          <Heatmap title="Forward code reconstruction" rows={grid(reconstruction.plain,12)} min={0} max={1}
            unit={`RMSE ${metric(reconstruction.initial.rmse)}`} grayscale/>
          <Heatmap title="Input guided reconstruction" rows={grid(reconstruction.decoded,12)} min={0} max={1}
            unit={`RMSE ${metric(reconstruction.final.rmse)}`} grayscale/>
        </div>
      </section>
    </>}
    {!home &&    <>
    <p className="hint">Site {site+1} · eccentricity {metric(sites[site].eccentricity)} · cross mode drive gain {metric(gate)} · instantaneous frequency shift {metric(shiftHz)} Hz</p>
    <div className="plot-grid">
      <InputPreview rgb={frames[frame]}/>
      <Heatmap title="Forward code reconstruction" rows={grid(reconstruction.plain, 12)} min={0} max={1}
        unit={`RMSE ${metric(reconstruction.initial.rmse)}`} grayscale/>
      {waveform}{activity}
      <Heatmap title="Population amplitude" rows={grid(magnitude)} min={0} max={.25} unit="0–0.25 a.u. display scale"/>
      <Heatmap title="Population phase" rows={grid(phase)} min={-Math.PI} max={Math.PI} unit="−π to π rad · gray if undefined"/>
      {(home || view === 'analysis') && <>
      <figure className="cove-figure"><figcaption>Retinotopic luminance<span>Central and peripheral sampling</span></figcaption>
        <svg viewBox="0 0 64 64" role="img" aria-label="Retinotopic sites positioned in image coordinates">
          {sites.map((point, i) => <circle key={i} cx={point.x+.5} cy={point.y+.5}
            r={i === site ? 2.4 : 1.55} fill={`hsl(0 0% ${100*row.brightness[i]}%)`}
            stroke={i === site ? '#60caff' : '#8a98a5'} strokeWidth={i === site ? .85 : .3}>
            <title>{`Site ${i+1}, luminance ${metric(row.brightness[i])}`}</title>
          </circle>)}
        </svg>
      </figure>
      <Heatmap title="Local contrast" rows={grid(row.contrast)} min={0} max={.5} unit="0–0.5"/>
      <Heatmap title="Retinotopic orientation drive" rows={grid(row.drive.map(modes => Math.hypot(...modes[mode])))}
        min={0} max={1} unit="0–1 a.u."/>
      <Heatmap title="Fourier spatial bands" rows={[row.spectrum]} min={0}
        max={Math.max(.01, ...row.spectrum)} unit="RMS · low → high"/>
      <Heatmap title="Temporal Haar detail" rows={grid(row.wavelets.map(v => v[scale]))}
        min={-.25} max={.25} unit={frames.length === 1 ? 'Still input · zero detail' : 'Signed luminance change'}/>
      <Heatmap title="Incoming site weights" rows={grid(sites.map((_, i) =>
        row.weights[site].find(edge => edge.j === i)?.weight ?? 0))} min={0} max={1} unit="0–1"/>
      <Trace title="Cross mode frequency shift" values={crossTrace.frequency} unit="Relative to nominal mode · Hz"
        start={firstTime} end={lastTime} cursor={sample}/>
      <Trace title="Phase and amplitude drive gain" values={crossTrace.gain} unit="Forcing multiplier"
        start={firstTime} end={lastTime} extent={[.82, 1]} cursor={sample}/>
      <Trace title="Modeled site waveform" values={signal} unit="Summed real state · a.u."
        start={`${metric(1/internalRate)} s`} end={`${metric(signal.length/internalRate)} s`}/>
      {spectrum.length > 0 && <Trace title="Modeled site frequency spectrum" values={spectrum} unit="Mean window power · dB"
        start="0 Hz" end={`${metric(upperHz)} Hz`}/>}
      <Heatmap title="Modeled site spectrogram" rows={spectrogramRows} min={spectrumPeak-50} max={spectrumPeak}
        unit={`0–${metric(Math.max(0, upperHz))} Hz · time → · dB`}/>
      <Heatmap title="Input guided reconstruction" rows={grid(reconstruction.decoded, 12)} min={0} max={1}
        unit={`RMSE ${metric(reconstruction.final.rmse)} · ${metric(reconstruction.final.psnr)} dB PSNR`} grayscale/>
      </>}
      {prediction.legacy.length > 0 && <Heatmap title="Legacy calibrated reconstruction" rows={grid(prediction.legacy, 12)}
        min={0} max={1} unit={`RMSE ${metric(imageError(row.pixels, prediction.legacy).rmse)}`} grayscale/>}
      {activeModel && <Trace title={`Predicted ERP · ${activeModel.subject} · ${activeModel.channels[electrode]}`}
        values={erp} unit="Scaled CSD" start="10 ms" end="590 ms" extent={[-peak, peak]}/>}
    </div>
    <p className="hint">The waveform cursor and activity maps show computed subsamples. Reconstruction updates per input frame.
      {(home || view === 'analysis') && ' Inverse refinement uses the input pixels and spans up to eight frames.'}</p>
    {activeModel?.schema === 'cove-erp-v2' && <p className="hint">
      Held-out calibration ({activeModel.test_conditions} images) · RMSE {metric(activeModel.held_out.rmse)} scaled CSD ·
      {' '}{metric(100*activeModel.held_out.skill)}% MSE skill vs training mean · {activeModel.mode.replaceAll('_', ' ')}
    </p>}
    {activeModel && <p className="hint">ERP predictions are relative to still-image onset. Held-out metrics describe the calibration dataset.</p>}
    {!home && stillImage && !model && <p className="hint">Load an exported calibration to display the participant-specific ERP prediction.</p>}
    </>}
  </section>;
}

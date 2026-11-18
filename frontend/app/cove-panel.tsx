'use client';

import { useMemo, useRef, useState } from 'react';
import { analyzeCove, flattenCode, GRID, HAAR_SCALES, MODES, sites, spectrogram, spectrumWindowLength } from '../lib/cove';
import { affine, type Calibration, ERP_BINS, parseCalibration, predictErp } from '../lib/cove-calibration';
import { imageError, refineCode } from '../lib/cove-reconstruction';
import renderer from '../lib/cove-decoder.json';

const DEFAULT_SLOPES = [2, 2];
const metric = (value: number) => Number(value.toPrecision(3)).toString();

function Heatmap({title, rows, min, max, unit, grayscale = false}: {
  title: string; rows: number[][]; min: number; max: number; unit: string; grayscale?: boolean;
}) {
  const height = rows.length, width = rows[0]?.length ?? 0;
  return <figure className="cove-figure">
    <figcaption>{title}<span>{unit}</span></figcaption>
    {height ? <svg viewBox="0 0 210 130" role="img" aria-label={title} preserveAspectRatio="none">
      {rows.flatMap((row, y) => row.map((value, x) => {
        const intensity = Math.min(1, Math.max(0, (value-min)/(max-min || 1)));
        const fill = grayscale ? `hsl(0 0% ${100*intensity}%)` :
          `hsl(${195 - 155*intensity} 65% ${12+65*intensity}%)`;
        return <rect key={y*width+x} x={210*x/width} y={130*y/height} width={210/width+.1} height={130/height+.1}
          fill={fill}>
          <title>{`Row ${y+1}, column ${x+1}: ${value.toPrecision(4)}`}</title>
        </rect>;
      }))}
    </svg> : <p className="hint">Insufficient samples</p>}
  </figure>;
}

function Trace({title, values, unit, start, end, extent}: {
  title: string; values: number[]; unit: string; start: string; end: string; extent?: [number, number];
}) {
  const [lo, hi] = extent ?? [Math.min(0, ...values), Math.max(0, ...values)];
  const span = Math.max(1e-12, hi-lo);
  return <figure className="cove-figure"><figcaption>{title}<span>{unit}</span></figcaption>
    <svg viewBox="0 0 210 130" role="img" aria-label={title}>
      <path className="axes" d="M29 12V105H200"/>
      <polyline className="trace t0" points={values.map((v, i) =>
        `${29+i*171/Math.max(1, values.length-1)},${105-93*(v-lo)/span}`).join(' ')}/>
      <text x="27" y="16" textAnchor="end">{metric(hi)}</text>
      <text x="27" y="105" textAnchor="end">{metric(lo)}</text>
      <text x="29" y="124">{start}</text><text x="200" y="124" textAnchor="end">{end}</text>
    </svg>
  </figure>;
}

function grid(values: number[], width = GRID) {
  return Array.from({length: width}, (_, row) => values.slice(row * width, (row + 1) * width));
}

export default function CovePanel({frames, frame, fps}: {frames: Uint8Array[]; frame: number; fps: number}) {
  const [model, setModel] = useState<Calibration>();
  const [error, setError] = useState('');
  const [site, setSite] = useState(12);
  const [mode, setMode] = useState(0);
  const [scale, setScale] = useState(0);
  const [electrode, setElectrode] = useState(0);
  const fileVersion = useRef(0);
  const activeModel = frames.length === 1 ? model : undefined;
  const slopes = activeModel?.coupling ?? DEFAULT_SLOPES;
  const {rows, internal, internalRate} = useMemo(() => analyzeCove(frames, fps, slopes, true), [frames, fps, slopes]);
  const row = rows[frame];
  const windowStart = Math.floor(frame / 8) * 8;
  const reconstruction = useMemo(() => {
    const window = rows.slice(windowStart, windowStart + 8);
    const forward = window.map(r => flattenCode(r.state));
    const refined = refineCode(forward, window.map(r => r.pixels), window.map(r => r.weights), renderer);
    return window.map((r, i) => {
      const plain = affine(forward[i], renderer.weights, renderer.offset);
      const decoded = affine(refined[i], renderer.weights, renderer.offset);
      return {plain, decoded, initial: imageError(r.pixels, plain), final: imageError(r.pixels, decoded)};
    });
  }, [rows, windowStart])[frame - windowStart];
  const {signal, spectrogramRows, spectrum, upperHz} = useMemo(() => {
    const signal = internal.map(sample => sample[site]);
    const power = spectrogram(signal, internalRate);
    const db = (value: number) => 10 * Math.log10(Math.max(1e-12, value));
    const bins = power[0]?.length ?? 0;
    const spectrum = Array.from({length: bins}, (_, k) => db(power.reduce((s, w) => s+w[k], 0) / power.length));
    const spectrogramRows = Array.from({length: bins}, (_, k) => power.map(w => db(w[bins-1-k])));
    const upperHz = (bins - 1) * internalRate / Math.max(1, spectrumWindowLength(signal.length, internalRate));
    return {signal, spectrogramRows, spectrum, upperHz};
  }, [internal, internalRate, site]);
  const spectrumPeak = Math.max(-119, ...spectrogramRows.flat());
  const magnitude = row.state.map(modes => Math.hypot(...modes[mode]));
  const phase = row.state.map(modes => Math.hypot(...modes[mode]) > 1e-9 ? Math.atan2(modes[mode][1], modes[mode][0]) : 0);
  const siteState = row.state[site][0];
  const gate = mode === 0 ? 1 : 1 - .18 * (1-siteState[0])/2;
  const shiftHz = 2.4 * mode * siteState[1] / (2*Math.PI);
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
  return <section className="cove-panel" aria-label="Distributed COVE analysis">
    <h3>Distributed COVE <small>· simulated population activity</small></h3>
    <div className="cove-controls">
      <label>Site <select value={site} onChange={event => setSite(Number(event.target.value))}>
        {sites.map((_, index) => <option key={index} value={index}>{index+1}</option>)}
      </select></label>
      <label>Mode <select value={mode} onChange={event => setMode(Number(event.target.value))}>
        {MODES.map((hz, index) => <option key={hz} value={index}>{hz} Hz</option>)}
      </select></label>
      <label>Haar scale <select value={scale} onChange={event => setScale(Number(event.target.value))}>
        {HAAR_SCALES.map((size, index) => <option key={size} value={index}>{size}+{size} frames</option>)}
      </select></label>
      {frames.length === 1 && <label>ERP calibration JSON <input type="file" accept=".json,application/json" onChange={async event => {
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
    <p className="hint">Site {site+1} · eccentricity {metric(sites[site].eccentricity)} · cross-mode drive gain {metric(gate)} · instantaneous frequency shift {metric(shiftHz)} Hz</p>
    <div className="plot-grid">
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
      <Heatmap title="Population amplitude" rows={grid(magnitude)} min={0} max={Math.max(.01, ...magnitude)} unit="Frame-scaled a.u."/>
      <Heatmap title="Population phase" rows={grid(phase)} min={-Math.PI} max={Math.PI} unit="−π to π rad"/>
      <Heatmap title="Incoming site weights" rows={grid(sites.map((_, i) =>
        row.weights[site].find(edge => edge.j === i)?.weight ?? 0))} min={0} max={1} unit="0–1"/>
      <Trace title="Modeled site waveform" values={signal} unit="Summed real state · a.u."
        start="0 s" end={`${metric((signal.length-1)/internalRate)} s`}/>
      {spectrum.length > 0 && <Trace title="Modeled site frequency spectrum" values={spectrum} unit="Mean window power · dB"
        start="0 Hz" end={`${metric(upperHz)} Hz`}/>}
      <Heatmap title="Modeled site spectrogram" rows={spectrogramRows} min={spectrumPeak-50} max={spectrumPeak}
        unit={`0–${metric(Math.max(0, upperHz))} Hz · time → · dB`}/>
      <Heatmap title="Forward code reconstruction" rows={grid(reconstruction.plain, 12)} min={0} max={1}
        unit={`RMSE ${metric(reconstruction.initial.rmse)}`} grayscale/>
      <Heatmap title="Inverse-refined reconstruction" rows={grid(reconstruction.decoded, 12)} min={0} max={1}
        unit={`RMSE ${metric(reconstruction.final.rmse)} · ${metric(reconstruction.final.psnr)} dB PSNR`} grayscale/>
      {prediction.legacy.length > 0 && <Heatmap title="Legacy calibrated reconstruction" rows={grid(prediction.legacy, 12)}
        min={0} max={1} unit={`RMSE ${metric(imageError(row.pixels, prediction.legacy).rmse)}`} grayscale/>}
      {activeModel && <Trace title={`Predicted ERP · ${activeModel.subject} · ${activeModel.channels[electrode]}`}
        values={erp} unit="Scaled CSD" start="10 ms" end="590 ms" extent={[-peak, peak]}/>}
    </div>
    <p className="hint">Reconstruction uses a procedural-data decoder. Inverse refinement uses the input pixels and spans up to eight frames.</p>
    {activeModel?.schema === 'cove-erp-v2' && <p className="hint">
      Held-out calibration ({activeModel.test_conditions} images) · RMSE {metric(activeModel.held_out.rmse)} scaled CSD ·
      {' '}{metric(100*activeModel.held_out.skill)}% MSE skill vs training mean · {activeModel.mode.replaceAll('_', ' ')}
    </p>}
    {activeModel && <p className="hint">ERP predictions are relative to still-image onset. Held-out metrics describe the calibration dataset.</p>}
    {frames.length === 1 && !model && <p className="hint">Load an exported calibration to display the participant-specific ERP prediction.</p>}
  </section>;
}

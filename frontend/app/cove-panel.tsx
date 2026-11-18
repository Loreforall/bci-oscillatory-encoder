'use client';

import { useMemo, useState } from 'react';
import { analyzeCove, GRID, MODES, sites, spectrogram } from '../lib/cove';

type Calibration = {
  schema: 'cove-erp-v1';
  subject: string;
  channels: string[];
  coupling: number[];
  weights: number[][];
  offset: number[];
  decoder: number[][];
  decoder_offset: number[];
};
const bins = 30;
const codeSize = GRID * GRID * MODES.length * 2;
const finiteVector = (value: unknown, length: number): value is number[] =>
  Array.isArray(value) && value.length === length &&
  value.every(n => typeof n === 'number' && Number.isFinite(n) && Math.abs(n) <= 1e6);

function calibrationFile(text: string): Calibration {
  const model = JSON.parse(text);
  const outputs = Array.isArray(model?.channels) ? model.channels.length * bins : 0;
  if (model?.schema !== 'cove-erp-v1' || !/^sub-0[12]$/.test(model.subject) ||
      outputs !== 240 || !model.channels.every((label: unknown) => typeof label === 'string' && /^[A-Za-z0-9]{1,5}$/.test(label)) ||
      !finiteVector(model.coupling, 2) || model.coupling.some((v: number) => v < 0 || v > 12) ||
      !Array.isArray(model.weights) || model.weights.length !== codeSize ||
      !model.weights.every((row: unknown) => finiteVector(row, outputs)) ||
      !finiteVector(model.offset, outputs) || !Array.isArray(model.decoder) ||
      model.decoder.length !== codeSize || !model.decoder.every((row: unknown) => finiteVector(row, 144)) ||
      !finiteVector(model.decoder_offset, 144)) {
    throw new Error('Select a cove-erp-v1 JSON file exported by the research calibration command.');
  }
  return model;
}

function Heatmap({title, rows, min, max, unit}: {
  title: string; rows: number[][]; min: number; max: number; unit: string;
}) {
  const height = rows.length, width = rows[0]?.length ?? 0;
  return <figure className="cove-figure">
    <figcaption>{title}<span>{unit}</span></figcaption>
    {height ? <svg viewBox="0 0 210 130" role="img" aria-label={title} preserveAspectRatio="none">
      {rows.flatMap((row, y) => row.map((value, x) => {
        const intensity = Math.min(1, Math.max(0, (value-min)/(max-min || 1)));
        return <rect key={y*width+x} x={210*x/width} y={130*y/height} width={210/width+.1} height={130/height+.1}
          fill={`hsl(${195 - 155*intensity} 65% ${12+65*intensity}%)`}>
          <title>{`Row ${y+1}, column ${x+1}: ${value.toPrecision(4)}`}</title>
        </rect>;
      }))}
    </svg> : <p className="hint">Insufficient samples</p>}
  </figure>;
}

function grid(values: number[]) {
  return Array.from({length: GRID}, (_, row) => values.slice(row * GRID, (row + 1) * GRID));
}

export default function CovePanel({frames, frame, fps}: {frames: Uint8Array[]; frame: number; fps: number}) {
  const [model, setModel] = useState<Calibration>();
  const [error, setError] = useState('');
  const [site, setSite] = useState(12);
  const [mode, setMode] = useState(0);
  const [electrode, setElectrode] = useState(0);
  const activeModel = frames.length === 1 ? model : undefined;
  const analysis = useMemo(() => analyzeCove(frames, fps, activeModel?.coupling), [frames, fps, activeModel]);
  const {rows, internal, internalRate} = analysis;
  const row = rows[frame];
  const spectrogramRows = useMemo(() => {
    const power = spectrogram(internal.map(sample => sample[site]), internalRate);
    return power[0]?.map((_, k) => power.map(column => 10 * Math.log10(Math.max(1e-12, column[k])))).reverse() ?? [];
  }, [internal, internalRate, site]);
  const spectrumPeak = Math.max(-120, ...spectrogramRows.flat());
  const magnitude = row.state.map(modes => Math.hypot(...modes[mode]));
  const phase = row.state.map(modes => Math.atan2(modes[mode][1], modes[mode][0]));
  const reconstruction = Array.from({length: 16}, (_, y) => Array.from({length: 16}, (_, x) => {
    const weights = sites.map(s => 1 / (1 + (s.x - 63*x/15)**2 + (s.y - 63*y/15)**2));
    return weights.reduce((sum, weight, i) => sum + weight*row.brightness[i], 0) /
      weights.reduce((sum, weight) => sum + weight, 0);
  }));
  const code = row.state.flatMap(modes => [...modes.map(pair => pair[0]), ...modes.map(pair => pair[1])]);
  const prediction = activeModel && model ? model.offset.slice(electrode*bins, (electrode+1)*bins).map((offset, bin) =>
    code.reduce((sum, value, i) => sum + value * model.weights[i][electrode*bins + bin], offset)) : [];
  const decoded = activeModel?.decoder_offset.map((offset, pixel) =>
    code.reduce((sum, value, i) => sum + value * activeModel.decoder[i][pixel], offset)) ?? [];
  const peak = Math.max(1e-9, ...prediction.map(Math.abs));
  return <section className="cove-panel" aria-label="Distributed COVE analysis">
    <h3>Distributed COVE <small>· simulated population activity</small></h3>
    <div className="cove-controls">
      <label>Site <select value={site} onChange={event => setSite(Number(event.target.value))}>
        {sites.map((_, index) => <option key={index} value={index}>{index+1}</option>)}
      </select></label>
      <label>Mode <select value={mode} onChange={event => setMode(Number(event.target.value))}>
        {MODES.map((hz, index) => <option key={hz} value={index}>{hz} Hz</option>)}
      </select></label>
      {frames.length === 1 && <label>ERP calibration JSON <input type="file" accept=".json,application/json" onChange={async event => {
        const file = event.target.files?.[0];
        setError('');
        setModel(undefined);
        if (!file) return;
        try {
          if (file.size > 4 * 1024**2) throw new Error('Calibration must be under 4 MiB.');
          setModel(calibrationFile(await file.text()));
          setElectrode(0);
        } catch (problem) {
          setError(problem instanceof Error ? problem.message : 'Calibration could not be loaded.');
        }
      }}/></label>}
      {activeModel && model && <label>Electrode <select value={electrode} onChange={event => setElectrode(Number(event.target.value))}>
        {model.channels.map((label, index) => <option key={label} value={index}>{label}</option>)}
      </select></label>}
    </div>
    {error && <p className="error" role="alert">{error}</p>}
    <div className="plot-grid">
      <Heatmap title="Local contrast" rows={grid(row.contrast)} min={0} max={.5} unit="0–0.5"/>
      <Heatmap title="Retinotopic input amplitude" rows={grid(row.drive.map(modes => Math.hypot(...modes[mode])))}
        min={0} max={1} unit="0–1 a.u."/>
      <Heatmap title="Population amplitude" rows={grid(magnitude)} min={0} max={Math.max(.01, ...magnitude)} unit="Frame-scaled a.u."/>
      <Heatmap title="Population phase" rows={grid(phase)} min={-Math.PI} max={Math.PI} unit="−π to π rad"/>
      <Heatmap title="Incoming site weights" rows={grid(sites.map((_, i) =>
        row.weights[site].find(edge => edge.j === i)?.weight ?? 0))} min={0} max={1} unit="0–1"/>
      <Heatmap title="Modeled site spectrogram" rows={spectrogramRows} min={spectrumPeak-50} max={spectrumPeak}
        unit={`0–${internalRate/2} Hz · time → · dB`}/>
      <Heatmap title="Foveated luminance reconstruction" rows={reconstruction} min={0} max={1} unit="0–1"/>
      {activeModel && <Heatmap title="Oscillatory code reconstruction" rows={Array.from({length: 12},
        (_, i) => decoded.slice(i*12, (i+1)*12))} min={0} max={1} unit="0–1"/>}
      {activeModel && model && <figure className="cove-figure"><figcaption>Predicted ERP · {model.subject} · {model.channels[electrode]}
        <span>{`±${peak.toPrecision(3)} scaled CSD`}</span></figcaption>
        <svg viewBox="0 0 210 130" role="img" aria-label="Predicted ERP from 10 to 590 ms">
          <path className="axes" d="M10 65H200"/>
          <polyline className="trace t0" points={prediction.map((v, i) => `${10+i*190/(bins-1)},${65-50*v/peak}`).join(' ')}/>
          <text x="10" y="125">10 ms</text><text x="166" y="125">590 ms</text>
        </svg>
      </figure>}
    </div>
    {activeModel && <p className="hint">Still-image pilot, 200 Hz CSD derivative. Predictions for new uploads are exploratory.</p>}
    {frames.length === 1 && !model && <p className="hint">Load a locally exported calibration to display an observational ERP prediction for a still image.</p>}
  </section>;
}

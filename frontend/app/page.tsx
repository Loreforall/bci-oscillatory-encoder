'use client';

import { FormEvent, useEffect, useState } from 'react';

const api = (process.env.NEXT_PUBLIC_ENCODER_URL ||
  (process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : '')).replace(/\/$/, '');
const email = process.env.NEXT_PUBLIC_RESEARCH_EMAIL;
const format = (n: number) => Number(n.toPrecision(5)).toString();

type Template = { band_power: number[][][]; band_phase: number[][][]; spatial_weights: number[][] };
type Result = {
  signal_type: string;
  input: {kind: string; width: number; height: number; frames: number; fps: number; processed_frames_sha256: string};
  parameters: {encoder: string; channels: number; bands_hz: number[]; sample_rate_hz: number; window_seconds: number; feature_size: number[]};
  feature_names: string[];
  features: number[][];
  motion_spectrum: {hz: number[]; power: number[]};
  template: Template;
  baseline: Template | null;
  waveforms: number[][][];
  metrics: {waveform_rms: number; peak_absolute: number; mean_band_power: number[]};
  comparison: Record<string, number> | null;
};
type Trace = { label: string; values: number[] };

function Plot({ title, traces, step, unit, range }: {
  title: string; traces: Trace[]; step: number; unit: string; range?: [number, number];
}) {
  const values = traces.flatMap(trace => trace.values);
  const min = range?.[0] ?? Math.min(...values, 0);
  const max = range?.[1] ?? Math.max(...values, 0);
  const last = Math.max(1, traces[0].values.length - 1);
  const x = (i: number) => 74 + i / last * 586;
  const y = (value: number) => 139 - (value - min) / (max - min || 1) * 119;
  return <figure>
    <figcaption>{title}</figcaption>
    <svg viewBox="0 0 682 173" role="img" aria-label={title}>
      {[0, 0.25, 0.5, 0.75, 1].map(t => <g key={t}>
        <path className="grid" d={`M74 ${20 + t * 119}H660 M${x(t * last)} 20V139`}/>
        <text x="68" y={24 + t * 119} textAnchor="end">{format(max - t * (max - min))}</text>
        <text x={x(t * last)} y="159" textAnchor="middle">{format(t * last * step)}</text>
      </g>)}
      {traces.map((trace, index) => trace.values.length === 1
        ? <circle key={trace.label} className={`trace t${index}`} cx={x(0)} cy={y(trace.values[0])} r="3"/>
        : <polyline key={trace.label} className={`trace t${index}`} points={trace.values.map((v, i) => `${x(i)},${y(v)}`).join(' ')}/>)}
    </svg>
    <div className="legend">{traces.map((trace, i) => <span className={`t${i}`} key={trace.label}>● {trace.label}</span>)}<span className="unit">{unit}</span></div>
  </figure>;
}

function save(filename: string, type: string, body: string) {
  const url = URL.createObjectURL(new Blob([body], {type}));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportCsv(result: Result) {
  const { input, parameters, template } = result;
  const rows = [['frame', 'frame_time_s', 'channel', 'spatial_weight', 'band_hz', 'band_power', 'phase_rad', ...result.feature_names].join(',')];
  template.band_power.forEach((frame, f) => frame.forEach((channel, c) => channel.forEach((power, b) => {
    rows.push([f, f / input.fps, c + 1, template.spatial_weights[f][c], parameters.bands_hz[b], power,
      template.band_phase[f][c][b], ...result.features[f]].join(','));
  })));
  save('oscillatory-template.csv', 'text/csv', rows.join('\n'));
}

export default function Workbench() {
  const [kind, setKind] = useState('synthetic');
  const [encoder, setEncoder] = useState('baseline');
  const [channels, setChannels] = useState(8);
  const [fps, setFps] = useState(24);
  const [sampleRate, setSampleRate] = useState(250);
  const [files, setFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<'checking' | 'ready' | 'offline'>('checking');
  const [learned, setLearned] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<Result>();
  const [channel, setChannel] = useState(0);
  const [band, setBand] = useState(0);
  const [frame, setFrame] = useState(0);
  const [feature, setFeature] = useState(0);

  async function check() {
    setStatus('checking');
    try {
      if (!api) throw new Error('Missing API URL');
      const response = await fetch(`${api}/health`, { cache: 'no-store', signal: AbortSignal.timeout(10000) });
      if (!response.ok) throw new Error('Unreachable');
      setLearned((await response.json()).learned === true);
      setStatus('ready');
    } catch {
      setStatus('offline');
    }
  }

  useEffect(() => { void check(); }, []);

  async function encode(event: FormEvent) {
    event.preventDefault();
    setError('');
    if (!api) { setError('The workbench API URL is missing.'); return; }
    if (kind !== 'synthetic' && (!files.length || files.length > 120 || files.reduce((n, file) => n + file.size, 0) >= 20 * 1024**2)) {
      setError('Select up to 120 images or one video, below 20 MiB total.');
      return;
    }
    const query = new URLSearchParams({kind, encoder, channels: String(encoder === 'learned' ? 8 : channels),
      fps: String(fps), sample_rate: String(sampleRate)});
    const body = new FormData();
    if (kind !== 'synthetic') files.forEach(file => body.append('files', file));
    setBusy(true);
    try {
      const response = await fetch(`${api}/encode?${query}`, {method: 'POST', body, signal: AbortSignal.timeout(90000)});
      if (!response.ok) {
        const problem = await response.json().catch(() => null);
        setError(typeof problem?.detail === 'string' ? problem.detail : `Encoding failed (HTTP ${response.status}).`);
        return;
      }
      setResult(await response.json());
      setChannel(0); setBand(0); setFrame(0); setFeature(0);
    } catch {
      setError('The encoder could not be reached or the request timed out.');
    } finally {
      setBusy(false);
    }
  }

  const step = result ? 1 / result.input.fps : 1;
  const bandTraces = (key: 'band_power' | 'band_phase'): Trace[] => {
    if (!result) return [];
    const label = `${result.parameters.bands_hz[band]} Hz`;
    const traces = [{label, values: result.template[key].map(row => row[channel][band])}];
    if (result.baseline) traces.push({label: 'Deterministic baseline', values: result.baseline[key].map(row => row[channel][band])});
    return traces;
  };
  return <>
    <header>
      <div><small>COMPUTATIONAL SIGNAL LAB</small><h1>BCI Oscillatory Encoder</h1></div>
      <nav aria-label="Sections"><a href="#workspace">Workspace</a><a href="#methods">Methods</a><a href="#research">Collaboration</a></nav>
      <button className="status" onClick={check} title="Check backend status"><i data-status={status}/>{status === 'ready' ? 'Engine online' : status === 'checking' ? 'Checking engine' : 'Engine offline'}</button>
    </header>
    <div className="notice">RESEARCH SIMULATION <span>Oscillatory templates are computational outputs. This workspace has no measured neural activity.</span></div>
    <main id="workspace">
      <aside>
        <h2>01 / Configuration</h2>
        <form onSubmit={encode}>
          <fieldset disabled={busy}>
            <legend>Input</legend>
            <label>Source<select value={kind} onChange={e => { setKind(e.target.value); setFiles([]); }}>
              <option value="synthetic">Synthetic moving grating</option><option value="video">Video upload</option><option value="images">Image sequence</option>
            </select></label>
            {kind !== 'synthetic' && <label>Media files<input key={kind} type="file" required multiple={kind === 'images'}
              accept={kind === 'images' ? 'image/png,image/jpeg,image/webp' : 'video/mp4,video/webm,video/quicktime,video/x-msvideo'}
              onChange={e => setFiles(Array.from(e.target.files ?? []))}/></label>}
            {kind === 'images' && <label>Sequence rate (frames/s)<input type="number" min="1" max="120" step="any" required value={fps} onChange={e => setFps(e.target.valueAsNumber)}/></label>}
            <p className="hint">{kind === 'synthetic' ? '48 frames · 128 × 96 px · 24 frames/s' :
              '20 MiB per request · 120 frames · 2.07 MP per frame. Image sequences follow filename order.'}</p>
          </fieldset>
          <fieldset disabled={busy}>
            <legend>Encoder</legend>
            <label>Model<select value={encoder} onChange={e => setEncoder(e.target.value)}>
              <option value="baseline">Deterministic baseline</option><option value="learned" disabled={!learned}>Learned GRU</option>
            </select></label>
            {!learned && <p className="hint">GRU requires a checkpoint that passes the held-out synthetic benchmark.</p>}
            <label>Channels<input type="number" min="1" max="16" required disabled={encoder === 'learned'} value={encoder === 'learned' ? 8 : channels} onChange={e => setChannels(e.target.valueAsNumber)}/></label>
            <label>Synthesis rate (Hz)<input type="number" min="100" max="500" step="1" required value={sampleRate} onChange={e => setSampleRate(e.target.valueAsNumber)}/></label>
            <p className="hint">Fixed bands · 6 / 10 / 20 / 40 Hz<br/>Window · 0.5 s per input frame</p>
          </fieldset>
          <button className="run" disabled={busy} type="submit">{busy ? 'Computing…' : 'Run encoder'}</button>
          {error && <p className="error" role="alert">{error}</p>}
        </form>
      </aside>
      <section className="visual">
        <h2>02 / Signal analysis <small>{result ? `${result.input.frames} FRAMES` : 'AWAITING RUN'}</small></h2>
        {result ? <>
          <div className="inspect">
            <label>Channel<select value={channel} onChange={e => setChannel(Number(e.target.value))}>
              {Array.from({length: result.parameters.channels}, (_, i) => <option key={i} value={i}>{i + 1}</option>)}
            </select></label>
            <label>Band<select value={band} onChange={e => setBand(Number(e.target.value))}>
              {result.parameters.bands_hz.map((hz, i) => <option key={hz} value={i}>{hz} Hz</option>)}
            </select></label>
            <label>Window<select value={frame} onChange={e => setFrame(Number(e.target.value))}>
              {result.features.map((_, i) => <option key={i} value={i}>Frame {i + 1}</option>)}
            </select></label>
            <label>Feature<select value={feature} onChange={e => setFeature(Number(e.target.value))}>
              {result.feature_names.map((name, i) => <option key={name} value={i}>{name.replaceAll('_', ' ')}</option>)}
            </select></label>
          </div>
          <Plot title="Visual feature trajectory" step={step} unit="Input time (s)" traces={[{label: result.feature_names[feature], values: result.features.map(row => row[feature])}]}/>
          <Plot title="Spatial FFT energy fractions" step={step} unit="Input time (s)" range={[0, 1]} traces={[
            {label: 'Low spatial frequencies', values: result.features.map(row => row[3])},
            {label: 'High spatial frequencies', values: result.features.map(row => row[4])}]}/>
          <Plot title="Temporal spectrum of frame motion" step={result.input.fps / result.input.frames} unit="Temporal frequency (Hz)" traces={[{label: 'Squared normalized FFT magnitude', values: result.motion_spectrum.power}]}/>
          <Plot title="Template band power" step={step} unit="Input time (s)" traces={bandTraces('band_power')}/>
          <Plot title="Wrapped phase (rad)" step={step} unit="Input time (s)" range={[-Math.PI, Math.PI]} traces={bandTraces('band_phase')}/>
          <Plot title="Synthesized waveform · all bands" step={1000 / result.parameters.sample_rate_hz} unit="Window time (ms)" traces={[{label: 'Simulated amplitude', values: result.waveforms[frame][channel]}]}/>
        </> : <div className="empty"><div className="crosshair">⊹</div><h3>No run loaded</h3><p>Run the built-in moving grating or upload media to inspect computed signals.</p></div>}
      </section>
      <aside>
        <h2>03 / Run inspection</h2>
        {result ? <>
          <h3>Input</h3><dl>
            <dt>Source</dt><dd>{result.input.kind}</dd><dt>Resolution</dt><dd>{result.input.width} × {result.input.height}</dd>
            <dt>Frames</dt><dd>{result.input.frames}</dd><dt>Frame rate</dt><dd>{format(result.input.fps)} fps</dd>
          </dl>
          <h3>Parameters</h3><dl>
            <dt>Encoder</dt><dd>{result.parameters.encoder}</dd><dt>Channels</dt><dd>{result.parameters.channels}</dd>
            <dt>Synthesis rate</dt><dd>{result.parameters.sample_rate_hz} Hz</dd><dt>Window</dt><dd>{result.parameters.window_seconds} s</dd>
            <dt>Feature grid</dt><dd>{result.parameters.feature_size.join(' × ')}</dd>
          </dl>
          <h3>Metrics</h3><dl>
            <dt>Waveform RMS</dt><dd>{format(result.metrics.waveform_rms)}</dd>
            <dt>Absolute peak</dt><dd>{format(result.metrics.peak_absolute)}</dd>
            <dt>Mean power · {result.parameters.bands_hz[band]} Hz</dt><dd>{format(result.metrics.mean_band_power[band])}</dd>
            <dt>Selected spatial weight</dt><dd>{format(result.template.spatial_weights[frame][channel])}</dd>
            {result.comparison && Object.entries(result.comparison).map(([name, value]) =>
              <div className="metric" key={name}><dt>{name.replaceAll('_', ' ')}</dt><dd>{format(value)}</dd></div>)}
          </dl>
          {result.comparison && <p className="hint">Comparison metrics use the deterministic template as reference.</p>}
          <h3>Export</h3><div className="exports">
            <button onClick={() => exportCsv(result)}>Template CSV</button>
            <button onClick={() => save('oscillatory-run.json', 'application/json', JSON.stringify(result))}>Full run JSON</button>
          </div>
          <p className="hint">JSON includes waveforms, engine versions and the processed input hash.</p>
        </> : <p className="hint">Metadata, metrics and exports appear after a successful run.</p>}
      </aside>
    </main>
    <article id="methods">
      <h2>Methods</h2>
      <p>This experiment asks how visual sequences can be represented as compact, multiband oscillatory templates for computational analysis.</p>
      <h3>Visual features</h3><p>Each frame is resized to 64 × 64 and converted to grayscale. Eight measurements describe luminance, contrast, Canny edge density, low and high non-DC spatial FFT energy, center and peripheral luminance, and mean absolute frame difference. Three mean-centered, L1-normalized cosine–Gaussian kernels at scales 1, 2 and 3 produce eight temporal features by averaging the absolute filter responses. The workspace also displays the squared normalized FFT magnitude of the centered motion trajectory, using the input frame rate as the timebase.</p>
      <h3>Deterministic mapping</h3><p>The 6, 10, 20 and 40 Hz bands use motion, brightness, contrast and the mean of edge density and high-frequency energy as respective power drivers. Each driver is scaled by 0.6, offset by 0.15 and multiplied by a channel gain from 0.8 to 1.2. Phase maps bounded motion, brightness, contrast and edge density linearly to [−π, π]. Spatial weights normalize a center–periphery mixture across channels.</p>
      <h3>GRU and circular phase</h3><p>A 16-feature input projection with ReLU feeds a 64-unit GRU. Three heads emit nonnegative band power through softplus, normalized phase pairs converted to angles by atan2, and softmax channel weights. Phase error uses the mean of 1 − cos(Δφ) across the circular wrap. The held-out synthetic benchmark compares power MSE, circular phase error and spatial MSE against a constant predictor. An enabled checkpoint must improve all three metrics.</p>
      <h3>Synthesis and interpretation</h3><p>For each input frame and channel, synthesis sums Pᵦ sin(2πfᵦt + φᵦ) over the fixed bands and multiplies by the channel weight. Template power is used as the amplitude coefficient by convention. Each frame creates an independent window. Phase is circular within the representation, and waveform continuity across windows is unconstrained.</p>
      <p>Evaluation measures reproduction of a synthetic deterministic teacher. Visual-to-oscillatory representations have not been validated as a means of delivering visual perception to the human brain. The workbench is computational only and produces simulated oscillations in arbitrary units.</p>
    </article>
    <article id="research"><h2>Research and collaboration</h2>
      <p>This experimental framework is available for research exploration. The project welcomes technical evaluation, research collaboration and inquiries about potential nonexclusive software licensing.</p>
      {email && <p><a href={`mailto:${email}`}>Contact the research team</a></p>}
    </article>
  </>;
}

'use client';

import { FormEvent, useRef, useState } from 'react';
import { analyze, type Input, type Result } from '../lib/encoder';
import { readInput } from '../lib/media';
import CovePanel from './cove-panel';
import FramePreview from './frame-preview';

const email = process.env.NEXT_PUBLIC_RESEARCH_EMAIL;
const format = (n: number) => Number(n.toPrecision(5)).toString();

type Trace = { label: string; values: number[] };

function Plot({ title, traces = [], step = 1, unit, valueUnit, range }: {
  title: string; traces?: Trace[]; step?: number; unit: string; valueUnit: string; range?: [number, number];
}) {
  const values = traces.flatMap(trace => trace.values);
  const hasData = values.length > 0;
  const min = range?.[0] ?? Math.min(...values, 0);
  const max = range?.[1] ?? Math.max(...values, 0);
  const upper = max === min ? min + 1 : max;
  const last = Math.max(1, (traces[0]?.values.length ?? 0) - 1);
  const plotLeft = 64;
  const plotRight = 372;
  const tick = (value: number) => Number(value.toPrecision(3)).toString();
  const x = (i: number) => plotLeft + i / last * (plotRight - plotLeft);
  const y = (value: number) => 110 - (value - min) / (upper - min) * 98;
  return <figure>
    <figcaption>{title}<span>{valueUnit}</span></figcaption>
    <svg viewBox="0 0 396 151" role="img" aria-label={hasData ? title : `${title}, no data`}>
      {[0, 0.25, 0.5, 0.75, 1].map(t => <g key={t}>
        <path className="grid" d={`M${plotLeft} ${12 + t * 98}H${plotRight} M${x(t * last)} 12V110`}/>
        {hasData && <text x={plotLeft - 5} y={15 + t * 98} textAnchor="end">{tick(upper - t * (upper - min))}</text>}
        {hasData && (values.length !== 1 || t === 0) && <text x={x(t * last)} y="125" textAnchor="middle">{tick(t * last * step)}</text>}
      </g>)}
      <path className="axes" d={`M${plotLeft} 12V110H${plotRight}`}/>
      <text x="219" y="144" textAnchor="middle">{unit}</text>
      {!hasData && <text x="219" y="65" textAnchor="middle">No samples</text>}
      {traces.map((trace, index) => trace.values.length === 1
        ? <circle key={trace.label} className={`trace t${index}`} cx={x(0)} cy={y(trace.values[0])} r="3"/>
        : <polyline key={trace.label} className={`trace t${index}`} points={trace.values.map((v, i) => `${x(i)},${y(v)}`).join(' ')}/>)}
    </svg>
    {hasData && <div className="legend">{traces.map((trace, i) => <span className={`t${i}`} key={trace.label}>━ {trace.label}</span>)}</div>}
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
  const [kind, setKind] = useState<Input['kind']>('synthetic');
  const [channels, setChannels] = useState(8);
  const [fps, setFps] = useState(24);
  const [videoFps, setVideoFps] = useState(12);
  const [sampleRate, setSampleRate] = useState(250);
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [run, setRun] = useState<{ result: Result; frames: Uint8Array[]; elapsedMs: number; id: number }>();
  const [formVersion, setFormVersion] = useState(0);
  const methods = useRef<HTMLDialogElement>(null);
  const verification = useRef<HTMLDialogElement>(null);
  const { result } = run ?? {};
  const [channel, setChannel] = useState(0);
  const [band, setBand] = useState(0);
  const [frame, setFrame] = useState(0);
  const [feature, setFeature] = useState(0);
  const [showCove, setShowCove] = useState(false);

  async function encode(event: FormEvent) {
    event.preventDefault();
    setError('');
    setBusy(true);
    const start = performance.now();
    try {
      // Let the busy state paint before synchronous feature and waveform work.
      await new Promise<void>(resolve => setTimeout(resolve, 0));
      const { frames, input } = await readInput(kind, files, kind === 'video' ? videoFps : fps);
      const result = await analyze(frames, input, channels, sampleRate);
      const elapsedMs = performance.now() - start;
      setRun(previous => ({result, frames, elapsedMs, id: (previous?.id ?? 0) + 1}));
      setChannel(0); setBand(0); setFrame(0); setFeature(0);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Local processing failed.');
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setKind('synthetic'); setChannels(8); setFps(24); setVideoFps(12); setSampleRate(250);
    setShowCove(false); setFiles([]); setRun(undefined); setError(''); setFormVersion(version => version + 1);
    setChannel(0); setBand(0); setFrame(0); setFeature(0);
  }

  const warnings = result ? [
    ...(result.input.kind === 'video' ? ['Browser seek sampling can repeat or skip source frames.'] : []),
    ...(result.input.frames === 1 ? ['One frame provides no temporal variation.'] : []),
  ] : [];
  const step = result ? 1 / result.input.fps : 1;
  const bandTraces = (key: 'band_power' | 'band_phase'): Trace[] => result
    ? [{label: `${result.parameters.bands_hz[band]} Hz`, values: result.template[key].map(row => row[channel][band])}]
    : [];
  const exportJson = () => {
    if (run) save('oscillatory-run.json', 'application/json', JSON.stringify({
      ...run.result, diagnostics: {elapsed_ms: run.elapsedMs, warnings},
    }));
  };
  return <>
    <header className="toolbar">
      <h1>BCI Oscillatory Encoder</h1>
      <div className="tools" role="group" aria-label="Workspace tools">
        <button className="run" type="submit" form="encoder-form" disabled={busy}>{busy ? 'Running…' : 'Run'}</button>
        <button onClick={reset} disabled={busy}>Reset</button>
        <details className="export-menu"><summary>Export</summary><div>
          <button disabled={!result} onClick={() => result && exportCsv(result)}>Template CSV</button>
          <button disabled={!result} onClick={exportJson}>Full run JSON</button>
        </div></details>
        <button onClick={() => methods.current?.showModal()}>Methods</button>
        <button onClick={() => verification.current?.showModal()}>Verification</button>
      </div>
      <span className="engine">LOCAL BROWSER ENGINE</span>
    </header>
    <div className="statusbar" role="status">
      <strong>{busy ? 'Processing' : error ? 'Run failed' : result ? 'Run complete' : 'Ready'}</strong>
      <span>{result ? `${result.input.frames} frames loaded` : 'No run loaded'}</span>
      <span className="validity">Simulated oscillatory templates · no measured neural activity</span>
    </div>
    <main id="workspace">
      <aside>
        <h2>01 / Configuration</h2>
        <form id="encoder-form" key={formVersion} onSubmit={encode}>
          <fieldset disabled={busy}>
            <legend>Input</legend>
            <label>Source<select value={kind} onChange={e => { setKind(e.target.value as Input['kind']); setFiles([]); }}>
              <option value="synthetic">Synthetic moving grating</option><option value="video">Video upload</option><option value="images">Image sequence</option>
            </select></label>
            {kind !== 'synthetic' && <label>Media files<input key={kind} type="file" required multiple={kind === 'images'}
              accept={kind === 'images' ? 'image/png,image/jpeg,image/webp' : 'video/mp4,video/webm,video/quicktime'}
              onChange={e => setFiles(Array.from(e.target.files ?? []))}/></label>}
            {kind === 'images' && <label>Sequence rate (frames/s)<input type="number" min="1" max="120" step="any" required value={fps} onChange={e => setFps(e.target.valueAsNumber)}/></label>}
            {kind === 'video' && <label>Video sampling rate (frames/s)<input type="number" min="1" max="120" step="any" required value={videoFps} onChange={e => setVideoFps(e.target.valueAsNumber)}/></label>}
            <p className="hint">{kind === 'synthetic' ? '48 frames · 128 × 96 px · 24 frames/s' :
              '20 MiB total · 120 samples · 2.07 MP per frame. Media stays in this browser.'}</p>
            {kind === 'images' && <p className="hint">Image sequences follow filename order.</p>}
            {kind === 'video' && <p className="hint">Video is sampled at the selected rate. Supported codecs depend on your browser.</p>}
          </fieldset>
          <fieldset disabled={busy}>
            <legend>Encoder</legend>
            <label>Model<select defaultValue="baseline">
              <option value="baseline">Deterministic baseline</option><option value="learned" disabled>Learned GRU</option>
            </select></label>
            <p className="hint">Learned GRU and checkpoint comparisons are available in local Python research.</p>
            <label>Channels<input type="number" min="1" max="16" required value={channels} onChange={e => setChannels(e.target.valueAsNumber)}/></label>
            <label>Synthesis rate (Hz)<input type="number" min="100" max="500" step="1" required value={sampleRate} onChange={e => setSampleRate(e.target.valueAsNumber)}/></label>
            <p className="hint">Fixed bands · 6 / 10 / 20 / 40 Hz<br/>Window · 0.5 s per input frame</p>
          </fieldset>
          {error && <p className="error" role="alert">{error}</p>}
        </form>
      </aside>
      <section className="visual" aria-labelledby="analysis-title">
        <h2 id="analysis-title">02 / Signal analysis <small>{result ? `FRAME ${frame + 1} / ${result.input.frames}` : 'AWAITING RUN'}</small></h2>
        {run && <div className="cove-extension">
          <button type="button" aria-expanded={showCove} onClick={() => setShowCove(value => !value)}>
            {showCove ? 'Hide distributed COVE' : 'Inspect distributed COVE'}
          </button>
          {showCove && <CovePanel key={run.id} frames={run.frames} frame={frame} onFrameChange={setFrame} source={run.result.input}/>}
        </div>}
        <div className="plot-grid">
          <section className="preview" aria-label="Visual input preview">
            <h3>Visual input <span>64 × 64 RGB</span></h3>
            <div className="frame-view">{run ? <FramePreview rgb={run.frames[frame]}/> : <span>Run the encoder to inspect input frames</span>}</div>
            <label>Frame <input type="range" min="0" max={Math.max(1, (result?.input.frames ?? 1) - 1)} step="1"
              value={frame} disabled={!result || result.input.frames < 2} onChange={event => setFrame(event.target.valueAsNumber)}/>
              <output>{result ? `${frame + 1} / ${result.input.frames} · ${format(frame * step)} s` : '—'}</output>
            </label>
          </section>
          <Plot title="Synthesized waveform · all bands" step={result ? 1000 / result.parameters.sample_rate_hz : 1}
            unit="Window time (ms)" valueUnit="Amplitude (a.u.)" traces={result ? [{label: `Channel ${channel + 1}`, values: result.waveforms[frame][channel]}] : []}/>
          <Plot title="Template band power" step={step} unit="Input time (s)" valueUnit="Power coefficient (a.u.)" traces={bandTraces('band_power')}/>
          <Plot title="Wrapped phase" step={step} unit="Input time (s)" valueUnit="Phase (rad)" range={[-Math.PI, Math.PI]} traces={bandTraces('band_phase')}/>
          <Plot title="Spatial FFT energy" step={step} unit="Input time (s)" valueUnit="Energy fraction" range={[0, 1]} traces={result ? [
            {label: 'Low spatial', values: result.features.map(row => row[3])},
            {label: 'High spatial', values: result.features.map(row => row[4])}] : []}/>
          <Plot title="Temporal spectrum of frame motion" step={result ? result.input.fps / result.input.frames : 1}
            unit="Temporal frequency (Hz)" valueUnit="Normalized power (a.u.)" traces={result ? [{label: 'Motion', values: result.motion_spectrum.power}] : []}/>
          <Plot title="Visual feature trajectory" step={step} unit="Input time (s)" valueUnit="Normalized feature" traces={result ? [
            {label: result.feature_names[feature], values: result.features.map(row => row[feature])}] : []}/>
        </div>

      </section>
      <aside aria-labelledby="inspector-title">
        <h2 id="inspector-title">03 / Run inspection</h2>
        <section className="inspector-section">
          <h3>Run diagnostics</h3>
          <dl>
            <dt>Source</dt><dd>{result?.input.kind ?? '—'}</dd>
            <dt>Input dimensions</dt><dd>{result ? `${result.input.width} × ${result.input.height} px` : '—'}</dd>
            <dt>Frame count</dt><dd>{result?.input.frames ?? '—'}</dd>
            <dt>Analysis rate</dt><dd>{result ? `${format(result.input.fps)} fps` : '—'}</dd>
            <dt>Encoder mode</dt><dd>{result?.parameters.encoder ?? '—'}</dd>
            <dt>Synthesis rate</dt><dd>{result ? `${result.parameters.sample_rate_hz} Hz` : '—'}</dd>
            <dt>Elapsed processing</dt><dd>{run ? `${format(run.elapsedMs)} ms` : '—'}</dd>
          </dl>
          <p className="hint"><strong>Warnings</strong> · {error ? 'Latest attempt failed. See configuration.' : result ? warnings.join(' ') || 'None' : 'No run to evaluate'}</p>
        </section>
        <section className="inspector-section">
          <h3>Channel and band</h3>
          <fieldset className="inspect" disabled={!result}>
            <label>Channel<select value={channel} onChange={e => setChannel(Number(e.target.value))}>
              {Array.from({length: result?.parameters.channels ?? 1}, (_, i) => <option key={i} value={i}>{i + 1}</option>)}
            </select></label>
            <label>Band<select value={band} onChange={e => setBand(Number(e.target.value))}>
              {(result?.parameters.bands_hz ?? [6, 10, 20, 40]).map((hz, i) => <option key={hz} value={i}>{hz} Hz</option>)}
            </select></label>
            <label>Feature<select value={feature} onChange={e => setFeature(Number(e.target.value))}>
              {(result?.feature_names ?? ['brightness']).map((name, i) => <option key={name} value={i}>{name.replaceAll('_', ' ')}</option>)}
            </select></label>
          </fieldset>
          <dl>
            <dt>Frame power (a.u.)</dt><dd>{result ? format(result.template.band_power[frame][channel][band]) : '—'}</dd>
            <dt>Frame phase (rad)</dt><dd>{result ? format(result.template.band_phase[frame][channel][band]) : '—'}</dd>
            <dt>Spatial weight</dt><dd>{result ? format(result.template.spatial_weights[frame][channel]) : '—'}</dd>
          </dl>
        </section>
        <section className="inspector-section">
          <h3>Run summary</h3>
          <dl>
            <dt>Waveform RMS (a.u.)</dt><dd>{result ? format(result.metrics.waveform_rms) : '—'}</dd>
            <dt>Absolute peak (a.u.)</dt><dd>{result ? format(result.metrics.peak_absolute) : '—'}</dd>
            <dt>Mean band power (a.u.)</dt><dd>{result ? format(result.metrics.mean_band_power[band]) : '—'}</dd>
            <dt>Synthesis window</dt><dd>{result ? `${result.parameters.window_seconds} s` : '—'}</dd>
          </dl>
        </section>
        <section className="inspector-section">
          <h3>Verification</h3>
          <p className="hint">{result ? result.runtime.engine_revision : 'Reference: Python deterministic pipeline'}</p>
          <button onClick={() => verification.current?.showModal()}>Open verification</button>
        </section>
        <section className="inspector-section">
          <h3>Export</h3>
          <div className="exports">
            <button disabled={!result} onClick={() => result && exportCsv(result)}>Template CSV</button>
            <button disabled={!result} onClick={exportJson}>Full run JSON</button>
          </div>
        </section>
      </aside>
    </main>
    <dialog ref={verification} aria-labelledby="verification-title">
      <h2 id="verification-title">Verification <form method="dialog"><button>Close</button></form></h2>
      <div className="documentation">
        <h3>Python reference fixtures</h3>
        <p>The regression suite compares browser computations with NumPy, OpenCV, and Torch on shared RGB fixtures. It checks visual features, temporal spectra, band coefficients, phase, spatial weights, waveform samples, RMS and peak.</p>
        <dl><dt>Feature absolute tolerance</dt><dd>2 × 10⁻⁶</dd><dt>Template absolute tolerance</dt><dd>1 × 10⁻⁶</dd><dt>Waveform absolute tolerance</dt><dd>5 × 10⁻⁶</dd></dl>
        <p>These checks assess numerical agreement on fixed inputs. Decoding and resizing can vary by browser. The experiment does not validate a neural response.</p>
        <h3>Loaded run provenance</h3>
        {result ? <><p>{result.runtime.engine_revision} · {result.input.sampling}</p><p className="hash">Processed RGB SHA-256: {result.input.processed_frames_sha256}</p></> : <p>No run loaded.</p>}
      </div>
    </dialog>
    <dialog ref={methods} aria-labelledby="methods-title">
      <h2 id="methods-title">Methods <form method="dialog"><button>Close</button></form></h2>
      <div className="documentation">
      <p>This experiment asks how visual sequences can be represented as compact, multiband oscillatory templates for computational analysis.</p>
      <h3>Visual features</h3><p>The browser decodes and resizes each frame to 64 × 64 before grayscale conversion. Codec, color conversion, and resampling can differ from the Python OpenCV pipeline. Video analysis uses uniformly spaced seek times at the configured rate. The analysis rate describes those sample times. Eight measurements describe luminance, contrast, Canny edge density, low and high non-DC spatial FFT energy, center and peripheral luminance, and mean absolute frame difference. Three mean-centered, L1-normalized cosine–Gaussian kernels at scales 1, 2 and 3 produce eight temporal features by averaging the absolute filter responses. The workspace also displays the squared normalized FFT magnitude of the centered motion trajectory, using the input frame rate as the timebase.</p>
      <h3>Deterministic mapping</h3><p>The 6, 10, 20 and 40 Hz bands use motion, brightness, contrast and the mean of edge density and high-frequency energy as respective power drivers. Each driver is scaled by 0.6, offset by 0.15 and multiplied by a channel gain from 0.8 to 1.2. Phase maps bounded motion, brightness, contrast and edge density linearly to [−π, π]. Spatial weights normalize a center–periphery mixture across channels.</p>
      <h3>GRU and circular phase</h3><p>A 16-feature input projection with ReLU feeds a 64-unit GRU. Three heads emit nonnegative band power through softplus, normalized phase pairs converted to angles by atan2, and softmax channel weights. Phase error uses the mean of 1 − cos(Δφ) across the circular wrap. The held-out synthetic benchmark compares power MSE, circular phase error and spatial MSE against a constant predictor. This benchmark and checkpoint-based comparisons are available in the local Python implementation.</p>
      <h3>Synthesis and interpretation</h3><p>For each input frame and channel, synthesis sums Pᵦ sin(2πfᵦt + φᵦ) over the fixed bands and multiplies by the channel weight. Template power is used as the amplitude coefficient by convention. Each frame creates an independent window. Phase is circular within the representation, and waveform continuity across windows is unconstrained.</p>
      <p>Evaluation measures reproduction of a synthetic deterministic teacher. Visual-to-oscillatory representations have not been validated as a means of delivering visual perception to the human brain. The workbench is computational only and produces simulated oscillations in arbitrary units.</p>
      <h3>Research and collaboration</h3>
      <p>This experimental framework is available for research exploration. The project welcomes technical evaluation, research collaboration and inquiries about potential nonexclusive software licensing.</p>
      {email && <p><a href={`mailto:${email}`}>Contact the research team</a></p>}
      </div>
    </dialog>
  </>;
}

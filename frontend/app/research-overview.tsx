const erp = [
  {label:'Previous nonlinear visual baseline', values:[8.09, 10.47]},
  {label:'Visual only kernel', values:[10.76, 13.01]},
  {label:'Hybrid COVE readout', values:[10.75, 12.95]},
];
const reconstruction = [
  {label:'Natural images', values:[.2301,.1203]},
  {label:'Procedural video', values:[.1621,.0679]},
];

function Benchmark({title, labels, rows, ceiling, digits, suffix}: {
  title: string; labels: string[]; rows: typeof erp; ceiling: number; digits: number; suffix: string;
}) {
  return <figure className="research-benchmark">
    <figcaption>{title}</figcaption>
    <table><thead><tr><th scope="col">Method or dataset</th>{labels.map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.label}><th scope="row">{row.label}</th>{row.values.map((value, i) =>
        <td key={i}><span className="research-bar" aria-hidden="true" style={{width:`${value / ceiling * 100}%`}}/>
        <span>{value.toFixed(digits)}{suffix}</span></td>)}</tr>)}</tbody>
    </table>
  </figure>;
}

export default function ResearchOverview() {
  return <article className="research-overview" id="research" aria-labelledby="research-title">
    <h2 id="research-title">The research behind COVE</h2>
    <div className="research-text">
      <section><h3>Research overview</h3><p>COVE is my proposed computational framework for translating visual information into structured oscillatory representations. The central question is how a visual scene can drive a spatially distributed field that retains interpretable information about location, frequency, amplitude and phase as it evolves.</p></section>
      <section><h3>Research background</h3><p><cite>Encoding the World into Brain Frequencies: A Pathway for Immersive BCIs</cite> outlined a broader sensory encoding problem. <cite>Encoding the World into the Brain Frequencies: A Neural Vision Framework</cite> developed the visual pathway. COVE turns that proposal into a functioning image and video encoder with inspectable states and reproducible computational evaluations.</p></section>
      <section><h3>Research objective</h3><p>The longer term objective is to investigate visual information encoding for future brain computer interfaces and immersive neurotechnology. The present work builds a software testbed for those questions. Biological interpretation and hardware validation require separate experiments.</p></section>
      <section><h3>Implemented capabilities</h3><p>The encoder processes images and video through four modes at 6, 10, 20 and 40 Hz. The workbench offers a 25 site reference and computational extensions with up to 256 sites. The reported EEG and reconstruction benchmarks use the reference field. The browser synchronizes visual frames with waveforms and spatial activity. The workbench also supports visual reconstruction, numerical exports and optional estimates of still image event related potentials, or ERPs, from a participant calibration.</p></section>
    </div>
    <section><h3>Computational methodology</h3>
      <ol className="research-pipeline" aria-label="COVE processing architecture">
        <li>Visual input</li><li>Spatial and temporal features</li><li>Foveated representation</li>
        <li>Oscillatory field with interacting modes</li><li>Complex state trajectories</li>
      </ol>
      <div className="research-outputs"><span>Visual reconstruction</span><span>Observational EEG evaluation</span></div>
      <p>The visual front end captures orientation, contrast and temporal change. Central and peripheral samples drive a complex field with spatial interactions and modulation between modes. Reconstruction measures information retained by that representation. EEG evaluation uses a separate calibrated readout that can also use visual features.</p>
    </section>
    <section><h3>Mathematical foundations</h3><p>Fourier analysis summarizes spatial frequency content. Causal wavelet differences describe temporal change at several scales. A spatial graph organizes interactions between foveated sites, and numerical integration advances a bounded complex dynamical system across time. A state can be written as <var>z(t) = A(t)e<sup>iφ(t)</sup></var>, where <var>A</var> is amplitude and <var>φ</var> is phase. The waveform plots show the real component of the computed state. Regularized inverse reconstruction uses the field coordinates to estimate a grayscale image.</p></section>
    <section id="evaluation"><h3>Experimental evaluation</h3>
      <p>The THINGS EEG2 pilot used two participants and three splits with visual concepts held out. ERP skill is the percentage reduction in mean squared error relative to the training mean. The visual only kernel performed comparably to the hybrid, so an independent predictive contribution from the oscillatory field remains unresolved.</p>
      <div className="research-results">
        <Benchmark title="Held out ERP MSE skill · higher is better" labels={['Participant 1','Participant 2']} rows={erp} ceiling={15} digits={2} suffix="%"/>
        <Benchmark title="Grayscale reconstruction RMSE · lower is better" labels={['Forward code','Input guided']} rows={reconstruction} ceiling={.25} digits={4} suffix=""/>
      </div>
      <p>The forward reconstruction uses the encoded state. Input guided refinement also uses the original visual input. Both scores refer to low resolution grayscale targets.</p>
    </section>
    <div className="research-text">
      <section><h3>Engineering verification</h3><p>The field update has a contraction bound for the same input sequence. Python and TypeScript parity tests compare visual features, complex trajectories, reconstruction and exports. The benchmark and decoder outputs reproduce in the reference environment. The browser workbench tests image and video input and the synchronization of playback with computed samples.</p></section>
      <section><h3>Research applications and future development</h3><p>The framework can support comparisons of visual representations, replication on larger EEG datasets and research on participant calibration. Potential hardware collaboration could begin with synchronized recording and data exchange, followed by separate evaluation of feedback and timing.</p></section>
      <section><h3>Scope and limitations</h3><p>COVE produces modeled oscillatory states. The EEG pilot evaluates observational prediction for still images. Generalization to continuous video EEG, live feedback and induction of visual perception have not been demonstrated. Spatial site maps display computed samples at their visual coordinates. Numerical exports in Advanced Analysis retain the reference field samples.</p></section>
    </div>
  </article>;
}

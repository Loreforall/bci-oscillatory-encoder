import { analyzeCove, crossMode, FOURIER_EDGES, GRID, HAAR_SCALES, MODES, sites } from './cove.ts';

type Analysis = ReturnType<typeof analyzeCove>;
type Source = {kind: string; fps: number; sampling: string; processed_frames_sha256: string};
const schema = 'cove-field-v1';

function siteCoordinates() {
  return sites.map(({x, y, eccentricity}, site) => ({
    site, row: Math.floor(site / GRID), column: site % GRID,
    x_normalized: 2*x/63-1, y_normalized: 2*y/63-1, eccentricity,
  }));
}

function frameRecord(row: Analysis['rows'][number], frame: number, analysis: Analysis, source: Source) {
  const states = row.state.map(modes => modes.map(([re, im]) => {
    const amplitude = Math.hypot(re, im);
    return {re, im, amplitude, phase_rad: amplitude > 1e-9 ? Math.atan2(im, re) : null};
  }));
  const relations = row.state.map((modes, i) => modes.map((mode, k) => {
    const {shiftHz, driveGain} = crossMode(modes[0], k);
    const hasPhase = states[i][k].phase_rad !== null && Math.hypot(...modes[0]) > 1e-9;
    return {source_mode: k === 0 ? null : 0, frequency_shift_hz: shiftHz, drive_gain: driveGain,
      relative_phase_rad: k > 0 && hasPhase
        ? Math.atan2(mode[1]*modes[0][0]-mode[0]*modes[0][1], mode[0]*modes[0][0]+mode[1]*modes[0][1]) : null};
  }));
  const graphEnergy = MODES.map((_, mode) => row.weights.reduce((sum, edges, i) => {
    const degree = 1 / edges.find(edge => edge.j === i)!.weight;
    const [r, q] = row.state[i][mode];
    return sum + .5 * degree * edges.reduce((value, edge) => value + edge.weight *
      ((r-row.state[edge.j][mode][0])**2 + (q-row.state[edge.j][mode][1])**2), 0);
  }, 0));
  return {
    frame, input_sample_time_s: frame/source.fps,
    state_time_s: (frame+1)*analysis.stepsPerFrame/analysis.internalRate,
    fourier_band_rms: row.spectrum, haar_detail: row.wavelets,
    luminance: row.brightness, contrast: row.contrast, motion_proxy: row.motion,
    conventional_features: row.conventional, quadrature_features: row.detail,
    pixels_12x12: row.pixels, complex_drive: row.drive, states,
    spatial_weights: row.weights.map(edges => edges.map(({j, weight}) => ({source_site:j, weight}))),
    cross_mode: relations, spatial_dirichlet_energy: graphEnergy,
  };
}

export function frameExport(analysis: Analysis, source: Source) {
  return analysis.rows.map((row, frame) => frameRecord(row, frame, analysis, source));
}

export function fieldJson(analysis: Analysis, source: Source, slopes: number[], calibration: unknown) {
  if (!analysis.field) throw new Error('Advanced field data is required');
  const width = sites.length * MODES.length * 2;
  const count = analysis.field.length / width;
  const header = {
    schema, signal_type: 'modeled_population_field', source,
    input_frame_count: analysis.rows.length, model_frame_rate_hz: analysis.modelFrameRate,
    integration_rate_hz: analysis.internalRate, steps_per_frame: analysis.stepsPerFrame,
    internal_first_time_s: 1 / analysis.internalRate,
    internal_complex_shape: [count, sites.length, MODES.length, 2],
    coordinates: siteCoordinates(), nominal_frequencies_hz: MODES,
    spatial_frequency_edges_cycles_per_pixel: FOURIER_EDGES, haar_half_windows_frames: HAAR_SCALES,
    spatial_slopes: slopes, frames: frameExport(analysis, source),
    observational_erp: calibration,
  };
  const parts = [JSON.stringify(header).slice(0, -1), ',"internal_complex_flat":['];
  // Serialize in bounded chunks without copying the entire double array into an object array.
  for (let i = 0; i < analysis.field.length; i += 16000) {
    if (i) parts.push(',');
    parts.push(analysis.field.subarray(i, i+16000).join(','));
  }
  parts.push(']}');
  return new Blob(parts, {type:'application/json'});
}

export function fieldCsv(analysis: Analysis, source: Source) {
  const frames = frameExport(analysis, source);
  const coordinates = siteCoordinates();
  const parts = ['frame,input_sample_time_s,state_time_s,site,row,column,x_normalized,y_normalized,eccentricity,mode,nominal_hz,real,imaginary,amplitude,phase_rad,source_mode,frequency_shift_hz,drive_gain,relative_phase_rad\n'];
  for (const record of frames) {
    parts.push(record.states.flatMap((modes, site) => modes.map((z, mode) => {
      const p = coordinates[site], c = record.cross_mode[site][mode];
      return [record.frame, record.input_sample_time_s, record.state_time_s, site, p.row, p.column,
        p.x_normalized, p.y_normalized, p.eccentricity, mode, MODES[mode], z.re, z.im, z.amplitude, z.phase_rad,
        c.source_mode, c.frequency_shift_hz, c.drive_gain, c.relative_phase_rad].join(',');
    })).join('\n') + '\n');
  }
  return new Blob(parts, {type:'text/csv;charset=utf-8'});
}

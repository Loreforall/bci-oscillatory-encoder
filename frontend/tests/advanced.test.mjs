import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { analyzeCove, flattenCode } from '../lib/cove.ts';
import { affine, parseCalibration, predictErp } from '../lib/cove-calibration.ts';
import { imageError, refineCode } from '../lib/cove-reconstruction.ts';

const load = url => JSON.parse(readFileSync(new URL(url, import.meta.url), 'utf8'));
const reference = load('./advanced-reference.json');
const renderer = load('../lib/cove-decoder.json');
const frames = Array.from({length: 5}, (_, t) => Uint8Array.from({length: 64**2 * 3}, (_, i) => {
  const p = Math.floor(i / 3), x = p % 64, y = Math.floor(p / 64);
  return [(13*x + 7*y + 17*t) % 256, (3*x + 11*y + 7*t) % 256,
    (5*x + 2*y + 29*t) % 256][i % 3];
}));
const {rows} = analyzeCove(frames, 24, [2, 0], true);

function close(actual, expected, tolerance) {
  assert.equal(actual.length, expected.length);
  assert.ok(Math.max(...actual.map((v, i) => Math.abs(v-expected[i]))) <= tolerance);
}

test('Fourier, causal Haar, complex dynamics and bypass agree with Python on RGB inputs', () => {
  rows.forEach((row, i) => close(row.spectrum, reference.spectrum[i], 2e-8));
  const row = rows[4];
  close(row.wavelets[12], reference.haar, 2e-8);
  close(flattenCode(row.state), reference.code, 2e-8);
  close(row.conventional, reference.feature, 2e-8);
  close(row.detail, reference.detail, 2e-8);
  close(row.pixels, reference.pixels, 2e-7);
});

test('inverse refinement agrees with Python and reduces image error', () => {
  const z = rows.map(row => flattenCode(row.state));
  const refined = refineCode(z, rows.map(row => row.pixels), rows.map(row => row.weights), renderer);
  const decoded = affine(refined[4], renderer.weights, renderer.offset);
  close(decoded, reference.reconstruction, 3e-7);
  assert.ok(imageError(rows[4].pixels, decoded).rmse <
    imageError(rows[4].pixels, affine(z[4], renderer.weights, renderer.offset)).rmse);
});

test('versioned calibration computes known ERP and rejects malformed feature blocks', () => {
  const offset = Array(240).fill(0.2);
  const mean = Array(rows[0].conventional.length).fill(0);
  const scale = Array(rows[0].conventional.length).fill(1);
  const model = {schema: 'cove-erp-v2', subject: 'sub-01',
    channels: ['O1','Oz','O2','PO7','PO3','POz','PO4','PO8'], coupling: [2,0],
    mode: 'visual_kernel', width: 2, mixture: [1,0,0], fit_conditions: 1, test_conditions: 40,
    blocks: [{mean, scale, anchors: [mean]}, null, null], held_out: {rmse:.1, skill:.1, skill_ci95:[0,.2]},
    offset, dual: [Array(240).fill(0.4)]};
  const values = predictErp(parseCalibration(JSON.stringify(model)), scale, [], []);
  close(values, Array(240).fill(.2+.4*Math.exp(-.5)), 1e-12);
  model.blocks[0].scale[249] = 0;
  assert.throws(() => parseCalibration(JSON.stringify(model)), /valid COVE ERP calibration/);
});

test('field exports preserve the full clock, complex samples, graph, and sparse phase', async () => {
  const {fieldJson, fieldCsv} = await import('../lib/cove-export.ts');
  const analysis = analyzeCove(frames, 24, [2, 0], true);
  const source = {kind:'images', fps:24, sampling:'filename-order', processed_frames_sha256:'a'.repeat(64)};
  const report = JSON.parse(await fieldJson(analysis, source, [2, 0], null).text());
  assert.equal(report.schema, 'cove-field-v1');
  assert.deepEqual(report.internal_complex_shape, [25, 25, 4, 2]);
  assert.equal(report.internal_first_time_s, 1/120);
  assert.equal(report.internal_complex_flat.length, 5000);
  const site12 = Array.from({length: 25}, (_, t) =>
    report.internal_complex_flat.slice(t*200+12*8, t*200+13*8)).flat();
  close(site12, reference.site_12_trace, 2e-8);
  report.frames.forEach((frame, t) => {
    assert.equal(frame.input_sample_time_s, t/24);
    assert.equal(frame.state_time_s, (t+1)/24);
    frame.spatial_weights.forEach(edges => close([edges.reduce((s,e)=>s+e.weight,0)], [1], 1e-12));
    assert.ok(frame.spatial_dirichlet_energy.every(energy => energy >= 0));
    const values = frame.states.flatMap(modes => modes.flatMap(({re,im})=>[re,im]));
    close(values, report.internal_complex_flat.slice((t*5+4)*200, (t+1)*5*200), 0);
  });
  const csv = (await fieldCsv(analysis, source).text()).trim().split('\n');
  assert.equal(csv.length, 501);
  assert.equal(csv[0].split(',').length, 19);
  assert.equal(csv[1].split(',').length, 19);
  const still = analyzeCove(frames.slice(0,1), 7, [2,0], true);
  const stillJson = JSON.parse(await fieldJson(still, {...source,fps:7}, [2,0], null).text());
  assert.equal(stillJson.model_frame_rate_hz, 24);
  assert.equal(stillJson.frames[0].state_time_s, 5/24);
});

test('playback follows the full-rate clock and the source-frame hold', async () => {
  const {playbackSample, playbackFrame} = await import('../lib/cove-playback.ts');
  const first = playbackSample(0, 41, 120, 240);
  assert.equal(first, 4);
  assert.equal(playbackFrame(first, 5), 0);
  assert.equal(playbackFrame(playbackSample(0, 42, 120, 240), 5), 1);
  assert.equal(playbackSample(6, 25, 120, 240), 9);
  assert.equal(playbackSample(6, 4000, 120, 240), 239);
  assert.equal(playbackFrame(24, 25), 0);
});

test('the landing example has reproducible, nonzero time-varying COVE output', async () => {
  const {coveExample, exampleInput} = await import('../lib/cove-example.ts');
  const media = coveExample();
  assert.equal(media.length, 48);
  assert.notDeepEqual(media[0], media[1]);
  const field = analyzeCove(media, exampleInput.fps, [2, 2], true);
  assert.equal(field.rows.length, media.length);
  assert.equal(field.internal.length, 240);
  assert.ok(field.field.some(value => Math.abs(value) > 0.01));
  assert.notDeepEqual(field.rows[0].state, field.rows[1].state);
});

test('the neighborhood photograph drives a held input response in all four modes', async () => {
  const {neighborhoodInput} = await import('../lib/cove-example.ts');
  const bytes = Buffer.from(JSON.parse(readFileSync(new URL('../lib/neighborhood-rgb.json', import.meta.url))), 'base64');
  assert.equal(bytes.length, 64 * 64 * 3);
  const {internal, field, stepsPerFrame, rows} = analyzeCove([bytes], neighborhoodInput.fps, [2, 2], true);
  assert.equal(stepsPerFrame, 25);
  assert.equal(internal.length, 25);
  assert.ok(rows[0].wavelets.flat().every(value => value === 0));
  for(let mode=0; mode<4; mode++) {
    const trace = internal.map((_, step) => field[(step*100+12*4+mode)*2]);
    assert.ok(Math.max(...trace)-Math.min(...trace)>1e-6);
  }
});

test('visualization projections preserve complex samples, graph weights and circular phase', async () => {
  const {modeTrajectory, spatialMatrix, phaseDifference} = await import('../lib/cove-visualization.ts');
  const {samplingTaps} = await import('../lib/cove.ts');
  const analysis = analyzeCove(frames, 24, [2,2], true);
  for (const site of [0,12,24]) {
    assert.ok(Math.abs(samplingTaps(site).reduce((sum,tap)=>sum+tap.weight,0)-1)<1e-12);
    const trace=modeTrajectory(analysis.field,site,2);
    assert.equal(trace.length,analysis.internal.length);
    analysis.rows.forEach((row,i)=>{
      const point=trace[(i+1)*analysis.stepsPerFrame-1];
      const [re,im]=row.state[site][2];
      assert.deepEqual([point.real,point.imaginary],[re,im]);
      assert.equal(point.amplitude,Math.hypot(re,im));
    });
  }
  const weights=analysis.rows[0].weights;
  const matrix=spatialMatrix(weights);
  assert.ok(matrix.every(row=>Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-12));
  weights.forEach((edges,i)=>edges.forEach(e=>assert.equal(matrix[i][e.j],e.weight)));
  assert.ok(Math.abs(phaseDifference(Math.PI-.1,-Math.PI+.1)+.2)<1e-12);
  assert.equal(phaseDifference(null,0),null);
});

test('expanded grids match Python at distributed integration samples and use normalized local weights', () => {
  const expected = JSON.parse(readFileSync(new URL('./expanded-reference.json', import.meta.url)));
  for (const side of [10,16]) {
    const result=analyzeCove(frames,24,[2,2],true,side);
    const n=side*side;
    assert.equal(result.sites.length,n);
    assert.equal(result.field.length,result.internal.length*n*8);
    assert.ok(result.field.every(Number.isFinite));
    const indices=[0,n/2,n-1];
    expected[String(side)].forEach((step,t)=>step.forEach((modes,j)=>modes.forEach((pair,k)=>{
      const offset=(t*n*4+indices[j]*4+k)*2;
      assert.ok(Math.abs(result.field[offset]-pair[0])<1e-8);
      assert.ok(Math.abs(result.field[offset+1]-pair[1])<1e-8);
    })));
    for(const row of result.rows) row.weights.forEach((edges,i)=>{
      assert.ok(edges.length>=3 && edges.length<=5);
      assert.ok(Math.abs(edges.reduce((sum,e)=>sum+e.weight,0)-1)<1e-12);
      assert.ok(edges.every(e=>Math.abs(i%side-e.j%side)+Math.abs(Math.floor(i/side)-Math.floor(e.j/side))<=1));
      assert.ok(row.state[i].every(([re,im])=>Math.hypot(re,im)<=1));
    });
  }
});

test('inverse Fourier display bands reconstruct the non DC image', async () => {
  const {spatialBand}=await import('../lib/features.ts');
  const {gray,FOURIER_EDGES}=await import('../lib/cove.ts');
  const input=gray(frames[0]), edges=[0,...FOURIER_EDGES,1];
  const bands=edges.slice(0,-1).map((v,i)=>spatialBand(input,v,edges[i+1]));
  const mean=input.reduce((a,b)=>a+b,0)/input.length;
  assert.ok(input.every((value,i)=>Math.abs(value-mean-bands.reduce((sum,band)=>sum+band[i],0))<1e-12));
});

test('overview and expanded racks use distinct computed channels and retain every integration sample', async () => {
  const {rackSites, rackState, fieldIndex} = await import('../lib/cove-visualization.ts');
  const {field, internalRate, internal, sites} = analyzeCove(frames, 24, [2, 0], true, 16);
  for (const expanded of [false, true]) {
    const selected = 119;
    const channels = rackSites(16, selected, expanded);
    assert.equal(new Set(channels).size, expanded ? 16 : 8);
    assert.ok(channels.includes(selected));
    for (const site of channels) for (let mode=0; mode<4; mode++) {
      const trajectory = [];
      for (let t=0; t<internal.length; t++) {
        const i = fieldIndex(t, site, mode, sites.length);
        assert.deepEqual(rackState(field, sites.length, site, mode, internalRate, t), {re:field[i], im:field[i+1]});
        const midpoint = rackState(field, sites.length, site, mode, internalRate, t-.5);
        assert.ok(Number.isFinite(midpoint.re) && Number.isFinite(midpoint.im));
        trajectory.push(field[i], field[i+1]);
      }
      assert.ok(trajectory.some(value => Math.abs(value) > 1e-9));
    }
  }
});

test('substep display follows damped rotation and preserves a zero input', async () => {
  const {rackState} = await import('../lib/cove-visualization.ts');
  const rate = 120, field = new Float64Array(16), theta = 2*Math.PI*40/rate;
  field[6] = 1;
  field[14] = Math.exp(-1/(.09*rate))*Math.cos(theta);
  field[15] = Math.exp(-1/(.09*rate))*Math.sin(theta);
  const midpoint = rackState(field, 1, 0, 3, rate, .5);
  close([midpoint.re,midpoint.im], [Math.cos(theta/2),Math.sin(theta/2)].map(v=>v*Math.exp(-.5/(.09*rate))), 1e-12);
  assert.deepEqual(rackState(new Float64Array(16),1,0,3,rate,.5), {re:0,im:0});
});

test('deterministic cortical populations respond to onset, contrast, orientation, video and spatial coupling', async () => {
  const {corticalResponse,CORTICAL_SUBSTEPS}=await import('../lib/cortical.ts');
  const image=analyzeCove([frames[0]],24,[2,2],true,16);
  const other=analyzeCove([frames[2]],24,[2,2],true,16);
  const {rows,internalRate,stepsPerFrame}=image;
  const run=(input=rows,options={})=>corticalResponse(input,internalRate,stepsPerFrame,{duration:.6,...options});
  const result=run();
  const width=16**2*8;
  assert.equal(result.rate,internalRate*CORTICAL_SUBSTEPS);
  assert.equal(result.field.length,result.count*width);
  assert.ok(result.field.subarray(0,result.latencySamples*width).every(v=>v===0));
  assert.ok(result.field.subarray(result.latencySamples*width).some(v=>Math.abs(v)>1e-4));
  assert.ok(result.field.every(v=>Number.isFinite(v)&&Math.abs(v)<15));
  assert.deepEqual(result.field,run().field);
  assert.notDeepEqual(result.field,run(other.rows).field);
  assert.notDeepEqual(result.field,run(rows,{lateral:false}).field);
  const changed=run([...rows,...other.rows]);
  const transition=(stepsPerFrame*CORTICAL_SUBSTEPS+result.latencySamples)*width;
  assert.deepEqual(result.field.subarray(0,transition),changed.field.subarray(0,transition));
  assert.notDeepEqual(result.field.subarray(transition),changed.field.subarray(transition));
  const blank=analyzeCove([new Uint8Array(frames[0].length)],24,[2,2],true,16);
  assert.ok(run(blank.rows).field.every(v=>v===0));
  const release=run(rows,{stimulusDuration:.1}).field;
  const rms=segment=>Math.sqrt(segment.reduce((sum,v)=>sum+v*v,0)/segment.length);
  assert.ok(rms(release.subarray(-width*20))<rms(release.subarray(width*20,width*40))*.15);
  const m=4*(16*6+6);
  const edge=rows[0].detail[m*4];
  const altered={...rows[0],detail:rows[0].detail.slice()};
  altered.detail[m*4]=edge+1;
  assert.notDeepEqual(result.field,run([altered]).field);
});

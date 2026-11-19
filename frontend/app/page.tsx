'use client';

import { useRef, useState } from 'react';
import CovePanel from './cove-panel';
import Workbench from './workbench';
import { coveExample, exampleInput } from '../lib/cove-example';
import type { Input } from '../lib/encoder';
import { readInput } from '../lib/media';

const example: {frames: Uint8Array[]; input: Input; label: string} = {frames: coveExample(), input: exampleInput, label: 'Moving grating example'};

export default function Home() {
  const [view, setView] = useState<'demo' | 'advanced'>('demo');
  const [media, setMedia] = useState(example);
  const [frame, setFrame] = useState(0);
  const [revision, setRevision] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const request = useRef(0);

  function replace(next: typeof example) {
    setMedia(next);
    setFrame(0);
    setRevision(value => value + 1);
  }

  async function upload(kind: Input['kind'], file: File | undefined) {
    if (!file) return;
    const version = ++request.current;
    setError('');
    setBusy(true);
    try {
      const loaded = await readInput(kind, [file], kind === 'video' ? 12 : 24);
      if (version === request.current) replace({...loaded, label: file.name});
    } catch (failure) {
      if (version === request.current) setError(failure instanceof Error ? failure.message : 'Could not read the selected file.');
    } finally {
      if (version === request.current) setBusy(false);
    }
  }

  return <>
    <header className="demo-nav">
      <span>COVE <small>Visual-to-oscillatory encoder</small></span>
      <nav aria-label="Workbench views">
        <button aria-current={view === 'demo' ? 'page' : undefined} onClick={() => setView('demo')}>Demonstration</button>
        <button aria-current={view === 'advanced' ? 'page' : undefined} onClick={() => setView('advanced')}>Advanced Analysis</button>
      </nav>
    </header>
    {view === 'advanced' ? <Workbench initialMedia={media}/> : <main className="demo-home">
      <div className="demo-intro">
        <div><h1>Visual input to oscillatory field</h1>
          <p>Follow how a moving image drives a field of 25 modeled sites and four oscillatory modes.</p>
        </div>
        <div className="demo-media">
          <label className="demo-upload">Upload image<input type="file" aria-label="Upload image" accept="image/png,image/jpeg,image/webp"
            onChange={event => {void upload('images', event.target.files?.[0]); event.target.value = '';}}/></label>
          <label className="demo-upload">Upload video<input type="file" aria-label="Upload video" accept="video/mp4,video/webm,video/quicktime"
            onChange={event => {void upload('video', event.target.files?.[0]); event.target.value = '';}}/></label>
          {media !== example && <button onClick={() => {
            request.current++;
            setBusy(false);
            setError('');
            replace(example);
          }}>Restore example</button>}
        </div>
      </div>
      {busy && <p role="status" className="demo-load">Reading media…</p>}
      {error && <p role="alert" className="error">{error}</p>}
      <CovePanel key={revision} frames={media.frames} frame={frame} onFrameChange={setFrame}
        source={media.input} home={{label: media.label, example: media === example}}/>
    </main>}
  </>;
}

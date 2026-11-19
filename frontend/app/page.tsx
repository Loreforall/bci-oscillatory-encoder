'use client';

import { useRef, useState } from 'react';
import CovePanel from './cove-panel';
import Workbench from './workbench';
import { neighborhoodInput } from '../lib/cove-example';
import type { Input } from '../lib/encoder';
import { readInput } from '../lib/media';
import rgb from '../lib/neighborhood-rgb.json';
import ResearchOverview from './research-overview';

const example: {frames: Uint8Array[]; input: Input; label: string} = {frames: [Uint8Array.from(atob(rgb), value => value.charCodeAt(0))], input: neighborhoodInput, label: 'Neighborhood photograph'};

export default function Home() {
  const [view, setView] = useState<'demo' | 'advanced'>('demo');
  const [media, setMedia] = useState(example);
  const [frame, setFrame] = useState(0);
  const [revision, setRevision] = useState(0);
  const [progress, setProgress] = useState('');
  const [dragging, setDragging] = useState(false);
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
    setProgress('Reading media…');
    try {
      const loaded = await readInput(kind, [file], kind === 'video' ? 12 : 24, (done,total) => {
        if (version === request.current) setProgress(`Reading sample ${done} of ${total}…`);
      });
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
        <div><p className="research-label">COVE RESEARCH WORKBENCH</p><h1>Visual input to oscillatory field</h1>
          <p>Explore how the photograph drives a simulated cortical response.</p></div>
        <div className="intro-links"><a href="#try-input">Try your own input ↓</a><a href="#research">Read the research ↓</a></div>
      </div>
      <CovePanel key={revision} frames={media.frames} frame={frame} onFrameChange={setFrame}
        source={media.input} home={{label: media.label, example: media === example, upload:
          <section className="demo-upload-section" aria-labelledby="try-input">
            <h2 id="try-input">Try Your Own Visual Input</h2>
            <div className={dragging ? "drop-area dragging" : "drop-area"}
              onDragOver={event => {event.preventDefault(); setDragging(true);}}
              onDragLeave={event => {if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false);}}
              onDrop={event => {
                event.preventDefault();
                setDragging(false);
                const file = event.dataTransfer.files[0];
                if (event.dataTransfer.files.length !== 1 || !file) {
                  setError('Drop one image or one video.');
                  return;
                }
                if (/\.(png|jpe?g|webp)$/i.test(file.name)) void upload('images', file);
                else if (/\.(mp4|webm|mov)$/i.test(file.name)) void upload('video', file);
                else setError('Choose a PNG, JPEG, WebP, MP4, WebM or MOV file.');
              }}>
              <div><strong>Drop an image or short video here</strong><p>PNG, JPEG or WebP · MP4, WebM or MOV · 20 MiB limit</p></div>
              <div className="demo-media">
                <label className="demo-upload">Upload image<input type="file" aria-label="Upload image" accept="image/png,image/jpeg,image/webp"
                  onChange={event => {void upload('images', event.target.files?.[0]); event.target.value = '';}}/></label>
                <label className="demo-upload">Upload video<input type="file" aria-label="Upload video" accept="video/mp4,video/webm,video/quicktime"
                  onChange={event => {void upload('video', event.target.files?.[0]); event.target.value = '';}}/></label>
                <button disabled={media === example && !busy} onClick={() => {
                  request.current++;
                  setBusy(false); setError(''); replace(example);
                }}>Restore photograph</button>
              </div>
            </div>
            {busy && <p role="status" className="demo-load">{progress}</p>}
            {error && <p role="alert" className="error">{error}</p>}
          </section>}}/>
      <ResearchOverview/>
    </main>}
  </>;
}

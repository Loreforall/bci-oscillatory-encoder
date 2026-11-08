import type { Input } from './encoder.ts';
import { SIZE } from './features.ts';

const MAX_FRAMES = 120;
const MAX_PIXELS = 1920 * 1080;
const MAX_BYTES = 20 * 1024 ** 2;

function validateSize(width: number, height: number) {
  if (!width || !height || width * height > MAX_PIXELS) {
    throw new Error('Choose media with at most 2,073,600 pixels per frame.');
  }
}

function frameReader() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = SIZE;
  const context = canvas.getContext('2d', {alpha: false, willReadFrequently: true});
  if (!context) throw new Error('This browser could not initialize image processing.');
  return (source: CanvasImageSource) => {
    // Use a black matte for transparent images, independently for each frame.
    context.clearRect(0, 0, SIZE, SIZE);
    context.drawImage(source, 0, 0, SIZE, SIZE);
    const rgba = context.getImageData(0, 0, SIZE, SIZE).data;
    const rgb = new Uint8Array(SIZE * SIZE * 3);
    for (let i = 0; i < SIZE * SIZE; i++) rgb.set(rgba.subarray(i * 4, i * 4 + 3), i * 3);
    return rgb;
  };
}

function waitForVideo(video: HTMLVideoElement, event: 'loadeddata' | 'seeked', start: () => void) {
  return new Promise<void>((resolve, reject) => {
    function cleanup() {
      clearTimeout(timeout);
      video.removeEventListener(event, done);
      video.removeEventListener('error', failed);
    }
    function done() { cleanup(); resolve(); }
    function failed() {
      cleanup();
      reject(new Error('The browser could not decode or seek this video. Try a short H.264 MP4 or WebM.'));
    }
    const timeout = setTimeout(failed, 8000);
    video.addEventListener(event, done);
    video.addEventListener('error', failed);
    start();
  });
}

export async function readInput(kind: Input['kind'], files: File[], fps: number) {
  const frames: Uint8Array[] = [];
  const readFrame = frameReader();
  if (kind === 'synthetic') {
    const canvas = document.createElement('canvas');
    const width = canvas.width = 128;
    const height = canvas.height = 96;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('This browser could not initialize the synthetic example.');
    const image = context.createImageData(width, height);
    for (let frame = 0; frame < 48; frame++) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const offset = (y * width + x) * 4;
          const value = Math.trunc(128 + 65 * Math.sin(x / 4 - frame / 3) * Math.cos(y / 8) + 25 * Math.sin(frame / 7));
          image.data[offset] = image.data[offset + 1] = image.data[offset + 2] = value;
          image.data[offset + 3] = 255;
        }
      }
      context.putImageData(image, 0, 0);
      frames.push(readFrame(canvas));
    }
    const input: Input = {kind, width, height, fps: 24, sampling: 'synthetic-grating'};
    return {frames, input};
  }

  if (!files.length || files.length > MAX_FRAMES || files.reduce((sum, file) => sum + file.size, 0) > MAX_BYTES) {
    throw new Error('Select up to 120 still images or one video, totaling at most 20 MiB.');
  }
  if (!Number.isFinite(fps) || fps < 1 || fps > 120) throw new Error('Choose a frame rate from 1 to 120.');
  if (kind === 'images') {
    let width = 0;
    let height = 0;
    const ordered = [...files].sort((a, b) => a.name < b.name ? -1 : Number(a.name > b.name));
    for (const file of ordered) {
      if (!/\.(png|jpe?g|webp)$/i.test(file.name)) throw new Error('Choose PNG, JPEG or WebP images.');
      const bitmap = await createImageBitmap(file).catch(() => {
        throw new Error('An image could not be decoded. Check that the selected files are valid still images.');
      });
      try {
        validateSize(bitmap.width, bitmap.height);
        if (width && (width !== bitmap.width || height !== bitmap.height)) {
          throw new Error('All images must have the same decoded resolution.');
        }
        width = bitmap.width;
        height = bitmap.height;
        frames.push(readFrame(bitmap));
      } finally {
        bitmap.close();
      }
    }
    const input: Input = {kind, width, height, fps, sampling: 'filename-order'};
    return {frames, input};
  }

  if (files.length !== 1) throw new Error('Select one video.');
  const video = document.createElement('video');
  const url = URL.createObjectURL(files[0]);
  video.preload = 'auto';
  video.muted = true;
  video.playsInline = true;
  try {
    await waitForVideo(video, 'loadeddata', () => { video.src = url; });
    const {videoWidth: width, videoHeight: height, duration} = video;
    validateSize(width, height);
    if (!Number.isFinite(duration) || duration <= 0) throw new Error('Choose a video with a finite duration.');
    const count = Math.ceil(duration * fps);
    if (count > MAX_FRAMES) throw new Error('Video would exceed 120 samples. Lower the sampling rate or trim the clip.');
    const deadline = performance.now() + 90000;
    for (let i = 0; i < count; i++) {
      if (performance.now() > deadline) throw new Error('Video sampling timed out. Try fewer samples or a smaller clip.');
      if (i > 0) await waitForVideo(video, 'seeked', () => { video.currentTime = i / fps; });
      if (video.videoWidth !== width || video.videoHeight !== height) {
        throw new Error('Video resolution must remain constant.');
      }
      frames.push(readFrame(video));
    }
    const input: Input = {kind, width, height, fps, sampling: 'uniform-browser-seek', duration_seconds: duration};
    return {frames, input};
  } finally {
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
  }
}

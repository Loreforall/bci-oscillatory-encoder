/** Convert the substep cursor into the input hold that produced that state. */
export function playbackFrame(sample: number, stepsPerFrame: number) {
  return Math.floor(sample / stepsPerFrame);
}

export function playbackSample(start: number, elapsedMs: number, rate: number, count: number) {
  return Math.min(count - 1, start + Math.floor(Math.max(0, elapsedMs) * rate / 1000));
}

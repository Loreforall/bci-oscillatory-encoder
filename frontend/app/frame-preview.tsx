import { useEffect, useRef } from 'react';
import { SIZE } from '../lib/features';

export default function FramePreview({rgb, label = "Processed RGB input frame"}: {rgb: Uint8Array; label?: string}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const image = context.createImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      image.data.set(rgb.subarray(i * 3, i * 3 + 3), i * 4);
      image.data[i * 4 + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [rgb]);
  return <canvas ref={canvas} width={SIZE} height={SIZE} role="img" aria-label={label}/>;
}

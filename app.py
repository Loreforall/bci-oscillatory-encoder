"""Public demonstration. Run: python app.py"""
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt
from baseline import encode_features, BAND_FREQUENCIES
from synthesizer import WaveformSynthesizer
from visual_features import VisualFeatureExtractor


def process(video_path):
    if not video_path:
        raise gr.Error('Upload a short video to begin')
    if Path(video_path).stat().st_size > 20 * 1024 * 1024:
        raise gr.Error('Video exceeds the 20 MB upload limit')
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        step = max(1, int(round(fps / 5)))
        i = 0
        while len(frames) < 60:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i += 1
    finally:
        cap.release()
    if not frames:
        raise gr.Error('Could not decode any frames')
    features = VisualFeatureExtractor().extract_sequence_features(frames)
    template = encode_features(features)
    tensor_template = {k: torch.from_numpy(v).unsqueeze(0) for k, v in template.items()}
    waveform = WaveformSynthesizer().synthesize(tensor_template)[0].detach().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    axes[0].plot(template['band_power'][:, 0, :])
    axes[0].legend([f'{int(f)} Hz' for f in BAND_FREQUENCIES], ncol=4)
    axes[0].set(xlabel='Sampled frame', ylabel='Relative amplitude', title='Band templates (channel 1)')
    axes[1].plot(np.arange(waveform.shape[-1]) / 250, waveform[0, 0])
    axes[1].set(xlabel='Seconds', ylabel='Arbitrary units', title='Synthesized waveform (first frame, channel 1)')
    output = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
    output.close()
    np.savez_compressed(output.name, features=features, **template, waveform=waveform,
                        band_frequencies=np.array(BAND_FREQUENCIES))
    return fig, output.name, f'Processed {len(frames)} frames. Synthetic oscillatory templates only; not EEG measurements or stimulation instructions.'

with gr.Blocks(title='Visual-to-Oscillatory Research Workbench') as demo:
    gr.Markdown('# Visual-to-Oscillatory Research Workbench\nComputational visual feature encoding demonstration. **Not validated for neural stimulation, diagnosis or neural response prediction.**')
    video = gr.Video(label='Upload a short video')
    run = gr.Button('Analyze')
    plot = gr.Plot(label='Signal representations')
    download = gr.File(label='Download numerical results (.npz)')
    status = gr.Textbox(label='Run information')
    run.click(process, inputs=video, outputs=[plot, download, status])

if __name__ == '__main__':
    demo.queue(max_size=8).launch(max_file_size='20mb')

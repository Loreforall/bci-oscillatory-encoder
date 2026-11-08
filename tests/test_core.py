import unittest
import numpy as np
import torch
from baseline import encode_features
from synthesizer import WaveformSynthesizer
from visual_features import VisualFeatureExtractor

class CoreTests(unittest.TestCase):
    def test_constant_frame(self):
        x = VisualFeatureExtractor().extract_sequence_features([np.zeros((64,64,3), dtype=np.uint8)])
        self.assertEqual(x.shape, (1,16))
        self.assertTrue(np.isfinite(x).all())
        self.assertGreaterEqual(x[0,4], 0)

    def test_template(self):
        out = encode_features(np.zeros((3,16), dtype=np.float32))
        self.assertEqual(out['band_power'].shape, (3,8,4))
        np.testing.assert_allclose(out['spatial_weights'].sum(axis=-1), 1, atol=1e-6)
        self.assertTrue(np.array_equal(out['band_power'], encode_features(np.zeros((3,16)))['band_power']))

    def test_waveform(self):
        template = {k:torch.tensor(v).unsqueeze(0) for k,v in encode_features(np.zeros((2,16))).items()}
        wave = WaveformSynthesizer().synthesize(template)
        self.assertEqual(tuple(wave.shape), (1,2,8,250))
        self.assertTrue(torch.isfinite(wave).all())

if __name__ == '__main__':
    unittest.main()

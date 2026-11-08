"""Research GRU for computational visual-to-oscillatory templates."""
import torch
from torch import nn
from torch.nn import functional as F

class SensoryOscillationEncoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, eeg_channels=8, bands=4):
        super().__init__()
        self.eeg_channels, self.bands = eeg_channels, bands
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.temporal_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.power_head = nn.Linear(hidden_dim, eeg_channels * bands)
        self.phase_head = nn.Linear(hidden_dim, eeg_channels * bands * 2)
        self.spatial_head = nn.Linear(hidden_dim, eeg_channels)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError('Expected [batch,time,features]')
        z, _ = self.temporal_encoder(F.relu(self.input_layer(x)))
        b, t, _ = z.shape
        power = F.softplus(self.power_head(z)).reshape(b,t,self.eeg_channels,self.bands)
        phase_vec = self.phase_head(z).reshape(b,t,self.eeg_channels,self.bands,2)
        phase_vec = F.normalize(phase_vec, dim=-1, eps=1e-6)
        phase = torch.atan2(phase_vec[...,0], phase_vec[...,1])
        spatial = torch.softmax(self.spatial_head(z), dim=-1)
        return {'band_power':power, 'band_phase':phase, 'spatial_weights':spatial}

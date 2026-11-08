"""Reproducible synthetic teacher-student benchmark, NOT neural validation.
Run: python train.py --epochs 80 --output artifacts/model.pt
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from baseline import encode_features
from model import SensoryOscillationEncoder

def make_batch(rng, batch=32, steps=20, channels=8):
    # Correlated, bounded synthetic visual features; targets are deterministic teacher outputs.
    x = rng.uniform(0.05, 0.95, (batch, steps, 16)).astype('float32')
    for t in range(1, steps):
        x[:,t] = 0.78*x[:,t-1] + 0.22*x[:,t]
    target = [encode_features(item, channels) for item in x]
    y = {k: torch.from_numpy(np.stack([item[k] for item in target])) for k in target[0]}
    return torch.from_numpy(x), y

def loss_parts(pred, target):
    power = F.mse_loss(pred['band_power'], target['band_power'])
    # Circular phase error remains well behaved at -pi / +pi.
    phase = (1 - torch.cos(pred['band_phase'] - target['band_phase'])).mean()
    spatial = F.mse_loss(pred['spatial_weights'], target['spatial_weights'])
    return power + 0.2*phase + spatial, {'power_mse':float(power.detach()), 'phase_circular':float(phase.detach()), 'spatial_mse':float(spatial.detach())}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--output', default='artifacts/model.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng_train = np.random.default_rng(args.seed)
    rng_val = np.random.default_rng(args.seed+1)
    model = SensoryOscillationEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    val_x, val_y = make_batch(rng_val, batch=64)
    best = float('inf')
    for epoch in range(args.epochs):
        model.train()
        x, y = make_batch(rng_train)
        pred = model(x)
        loss, _ = loss_parts(pred, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss, parts = loss_parts(model(val_x), val_y)
        if float(val_loss) < best:
            best = float(val_loss)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict':model.state_dict(),'seed':args.seed,'epoch':epoch,'validation':parts,'teacher':'deterministic_baseline_v1'},args.output)
        if epoch == 0 or (epoch+1)%10 == 0:
            print(f'epoch={epoch+1} train={float(loss):.5f} validation={float(val_loss):.5f} {parts}')
    print(f'best validation loss={best:.6f}; checkpoint={args.output}')

if __name__ == '__main__':
    main()

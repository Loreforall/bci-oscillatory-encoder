"""Evaluate GRU against deterministic teacher on a held-out synthetic set."""
import argparse
import numpy as np
import torch
from model import SensoryOscillationEncoder
from train import make_batch, loss_parts

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='artifacts/model.pt'); a=p.parse_args()
    # Only load checkpoints created locally by this repository, not untrusted files.
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=SensoryOscillationEncoder(); model.load_state_dict(checkpoint['state_dict']); model.eval()
    x,y=make_batch(np.random.default_rng(20261108),batch=128)
    with torch.no_grad():
        prediction=model(x); loss, metrics=loss_parts(prediction,y)
    # Simple constant predictor gives an independent lower-complexity reference.
    constant={k: v.mean(dim=(0,1),keepdim=True).expand_as(v) for k,v in y.items()}
    _, reference=loss_parts(constant,y)
    print('Held-out synthetic teacher reproduction (lower is better)')
    for key in metrics:
        print(f'{key}: GRU={metrics[key]:.6f} constant={reference[key]:.6f}')
    print('Not a biological EEG benchmark; baseline defines the synthetic targets.')
if __name__=='__main__': main()

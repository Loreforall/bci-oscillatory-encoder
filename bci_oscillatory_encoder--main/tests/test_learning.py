import numpy as np
import torch
from model import SensoryOscillationEncoder
from train import make_batch, loss_parts

def test_model_outputs_and_gradients():
    model=SensoryOscillationEncoder(hidden_dim=16)
    x,y=make_batch(np.random.default_rng(9),batch=2,steps=5)
    pred=model(x)
    assert pred['band_power'].shape == (2,5,8,4)
    assert torch.allclose(pred['spatial_weights'].sum(-1),torch.ones(2,5),atol=1e-5)
    loss,_=loss_parts(pred,y)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.input_layer.weight.grad is not None

def test_phase_wrap_loss():
    phase_a=torch.tensor([[[[np.pi-0.001]]]])
    phase_b=torch.tensor([[[[-np.pi+0.001]]]])
    assert (1-torch.cos(phase_a-phase_b)).mean() < 1e-4

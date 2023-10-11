import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
Array = np.ndarray


def t_to_sigma(
    t_tr: Tensor, t_rot: Tensor, 
    tr_sigma_min: float, tr_sigma_max: float, 
    rot_sigma_min: float, rot_sigma_max: float
) -> tuple[Tensor, Tensor]:
    """Translation and rotation times to corresponding variances."""
    tr_sigma = (tr_sigma_min ** (1-t_tr)) * (tr_sigma_max ** t_tr)
    rot_sigma = (rot_sigma_min ** (1-t_rot)) * (rot_sigma_max ** t_rot)
    return tr_sigma, rot_sigma


def get_t_schedule(inference_steps: int) -> Tensor:
    """Time schedule for reverse diffusion."""
    return torch.linspace(1, 0, inference_steps + 1, dtype=torch.float)[:-1]


def sinusoidal_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    max_positions: int = 10000
) -> Tensor:
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(
    embedding_type: str, 
    embedding_dim: int, 
    embedding_scale: int = 10000
) -> Callable:
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x : sinusoidal_embedding(embedding_scale * x, embedding_dim))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func

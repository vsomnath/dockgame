"""SO(3) score and sampling computation. 

Preprocessing for the SO(3) sampling and score computations, truncated infinite 
series are computed and then cached to memory, therefore the precomputation is 
only run the first time the repository is run on a machine.

Modified from https://github.com/gcorso/DiffDock/blob/main/utils/so3.py
"""
import os
from typing import Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Type aliases
Tensor = torch.Tensor
Array = np.ndarray
FloatOrArray = Union[float, Array]


MIN_EPS, MAX_EPS, N_EPS = 0.01, 2, 1000
X_N = 2000


omegas = np.linspace(0, np.pi, X_N + 1)[1:]


# R1 @ R2 but for Euler vecs
def _compose(r1: Array, r2: Array) -> Array:
    r1_mat = Rotation.from_rotvec(r1).as_matrix()
    r2_mat = Rotation.from_rotvec(r2).as_matrix()
    return Rotation.from_matrix(r1_mat @ r2_mat).as_rotvec()


 # the summation term only
def _expansion(omega: float, eps: float, L: int = 2000) -> Array: 
    p = 0
    for l in range(L):
        p += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2) * np.sin(omega * (l + 1 / 2)) / np.sin(omega / 2)
    return p


def _density(expansion, omega, marginal=True):
    if marginal:
        # if marginal, density over [0, pi], else over SO(3)
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        # the constant factor doesn't affect any actual calculations though
        return expansion / 8 / np.pi ** 2  


# score of density over SO(3)
def _score(exp: FloatOrArray, omega: float, eps: float, L: int = 2000):  
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2)
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2) * (lo * dhi - hi * dlo) / lo ** 2
    return dSigma / exp


if os.path.exists('.so3_omegas_array2.npy'):
    _omegas_array = np.load('.so3_omegas_array2.npy')
    _cdf_vals = np.load('.so3_cdf_vals2.npy')
    _score_norms = np.load('.so3_score_norms2.npy')
    _exp_score_norms = np.load('.so3_exp_score_norms2.npy')
else:
    print("Precomputing and saving to cache SO(3) distribution table")
    _eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
    _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

    _exp_vals = np.asarray([_expansion(_omegas_array, eps) for eps in _eps_array])
    _pdf_vals = np.asarray([_density(_exp, _omegas_array, marginal=True) for _exp in _exp_vals])
    _cdf_vals = np.asarray([_pdf.cumsum() / X_N * np.pi for _pdf in _pdf_vals])
    _score_norms = np.asarray([_score(_exp_vals[i], _omegas_array, _eps_array[i]) for i in range(len(_eps_array))])

    _exp_score_norms = np.sqrt(np.sum(_score_norms**2 * _pdf_vals, axis=1) / np.sum(_pdf_vals, axis=1) / np.pi)

    np.save('.so3_omegas_array2.npy', _omegas_array)
    np.save('.so3_cdf_vals2.npy', _cdf_vals)
    np.save('.so3_score_norms2.npy', _score_norms)
    np.save('.so3_exp_score_norms2.npy', _exp_score_norms)


def sample(eps: FloatOrArray) -> Array:
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    x = np.random.rand()
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array)


def sample_vec(eps: FloatOrArray) -> Array:
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * sample(eps)


def score_vec(eps: FloatOrArray, vec: Array) -> Array:
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    om = np.linalg.norm(vec)
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om


def score_norm(eps: FloatOrArray) -> Tensor:
    eps = eps.numpy()
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS-1)
    return torch.from_numpy(_exp_score_norms[eps_idx]).float()

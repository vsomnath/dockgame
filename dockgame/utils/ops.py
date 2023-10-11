from typing import Callable, Any, Union

import torch
import numpy as np

# Type aliases
Tensor = torch.Tensor
Array = np.ndarray

# ==============================================================================
# Tensor ops
# ==============================================================================


def rbf_basis(
        inputs: Tensor, 
        basis_min: float, 
        basis_max: float, 
        interval: int
    ) -> Tensor:
    """Expands the provided input tensor into a RBF basis.
    
    Args:
    inputs: The tensor we want to expand into the RBF basis
    basis_min: The smallest center in the RBF basis
    basis_max:  The largest center in the RBF basis
    interval: Number of centers used in the RBF basis

    Returns:
        torch.Tensor object expanded into radial basis
    """

    # Compute the centers of the basis
    n_basis_terms = int((basis_max - basis_min) / interval)
    mus = torch.linspace(basis_min, basis_max, n_basis_terms)
    mus = mus.view(1, -1) # [1, n_basis_terms]

    inputs_expanded = inputs.unsqueeze(dim=-1) # [n_residues, 1]

    # Shape: [n_residues, n_basis_terms]
    return torch.exp( -((inputs_expanded - mus) / interval)**2)


def to_numpy(tensor: Tensor) -> Array:
    """Simple wrapper to convert a torch.Tensor into np.ndarray.
    
    Args:
        tensor: the tensor object we want to conver to numpy
    
    Returns:
        np.ndarray
    """
    if tensor.device != "cpu":
        tensor = tensor.cpu()

    if tensor.requires_grad_:
        return tensor.detach().numpy()
    return tensor.numpy()
 

# ==============================================================================
# Featurizing ops
# ==============================================================================


def onek_encoding_unk(x: Any, allowable_set: Union[list, set]) -> list:
    """One-hot encoding of a given value against an allowable set (with UNK)."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


def index_with_unk(a, b):
    """Returns element value if present else value corresponding to unk"""
    if b in a:
        return a[b]
    else:
        return a['unk']


def _tri_flatten(
    tri: Tensor,  
    indicies_func: Callable, 
    offset: int = 1,
) -> Tensor:
    N = tri.size(-1)
    indicies = indicies_func(N, N, offset=offset) #Offset=1 because no diagonal needed
    indicies = N * indicies[0] + indicies[1]
    return tri.flatten(-2)[..., indicies]


def tril_flatten(tril: Tensor, offset: int = 1) -> Tensor:
    """Flatten upper on lower triangular matrix."""
    return _tri_flatten(tril, torch.tril_indices, offset=offset)


def triu_flatten(triu: Tensor, offset: int = 1) -> Tensor:
    """Flatten based on upper triangular matrix"""
    return _tri_flatten(triu, torch.triu_indices, offset=offset)


def make_float_tensor(array: Array) -> Tensor:
    return torch.tensor(array, dtype=torch.float32)

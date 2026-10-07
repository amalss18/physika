"""Tutorial-local bindings for the original vectorized tensor operations.

The .phyk implementations let the unchanged compiler type-check and compile
the equations. The runner binds their Torch equivalents in the model's own
namespace, preserving the original kernels and autodiff behavior. Nothing
in the Physika compiler or global runtime is patched.
"""
import torch


def compatible_tensors(a, b):
    if isinstance(a, torch.Tensor) and not isinstance(b, torch.Tensor):
        return a, torch.as_tensor(b, dtype=a.dtype, device=a.device)
    if isinstance(b, torch.Tensor) and not isinstance(a, torch.Tensor):
        return torch.as_tensor(a, dtype=b.dtype, device=b.device), b
    return torch.as_tensor(a), torch.as_tensor(b)


def minimum(a, b):
    a, b = compatible_tensors(a, b)
    dtype = torch.promote_types(a.dtype, b.dtype)
    device = a.device if a.device.type != 'cpu' else b.device
    return torch.minimum(a.to(dtype=dtype, device=device),
                         b.to(dtype=dtype, device=device))


def maximum(a, b):
    a, b = compatible_tensors(a, b)
    dtype = torch.promote_types(a.dtype, b.dtype)
    device = a.device if a.device.type != 'cpu' else b.device
    return torch.maximum(a.to(dtype=dtype, device=device),
                         b.to(dtype=dtype, device=device))


def softplus(a):
    return torch.nn.functional.softplus(torch.as_tensor(a))


def matrix_vector(matrix, vector, n):
    return torch.mv(matrix, vector)


def bind_tensor_operations(namespace):
    """Install elementary operations only in this compiled model's globals."""
    namespace.update(minimum=minimum, maximum=maximum, softplus=softplus,
                     sinh=torch.sinh, tanh=torch.tanh,
                     matrix_vector=matrix_vector)

from .utils import get_test_devices, init_device
from nitorch_fastmath.realtransforms import (
    dct, dctn, dst, dstn, idct, idctn, idst, idstn,
)
from torch.autograd import gradcheck
import inspect
import torch
import pytest

devices = get_test_devices()
functions = ('dct', 'dst', 'idct', 'idst')
types = (1, 2, 3)
norms = ('forward', 'backward', 'ortho', 'ortho_scipy')


# set gradcheck options
if hasattr(torch, 'use_deterministic_algorithms'):
    torch.use_deterministic_algorithms(True)
kwargs = dict(
    # rtol=1.,
    raise_exception=True,
    check_grad_dtypes=True,
)
if 'check_undefined_grad' in inspect.signature(gradcheck).parameters:
    kwargs['check_undefined_grad'] = False
if 'nondet_tol' in inspect.signature(gradcheck).parameters:
    kwargs['nondet_tol'] = float('inf')


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("function", functions)
@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
def test_dctdst_gradcheck(device, function, type, norm):
    device = init_device(device)
    dtype = torch.double
    backend = dict(dtype=dtype, device=device)

    if type == 4 and device.type != 'cpu':
        # Type IV not implemented on GPU
        return

    function1 = (dct if function == 'dct' else
                 dst if function == 'dst' else
                 idct if function == 'idct' else
                 idst if function == 'idst' else None)
    functionn = (dctn if function == 'dct' else
                 dstn if function == 'dst' else
                 idctn if function == 'idct' else
                 idstn if function == 'idst' else None)

    def check(x, dim=-1):
        fn = function1 if isinstance(dim, int) else functionn
        return gradcheck(fn, (x, dim, norm, type), **kwargs)

    mat = torch.randn([4, 5], **backend)
    mat.requires_grad = True
    assert check(mat, dim=-1), "dim=-1"
    assert check(mat, dim=0), "dim=0"
    assert check(mat, dim=None), "dim=all"
    assert check(mat, dim=[0, 1]), "dim=[0, 1]"

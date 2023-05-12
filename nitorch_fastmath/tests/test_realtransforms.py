from .utils import get_test_devices, init_device
from nitorch_fastmath.realtransforms import (
    dct, dctn, dst, dstn, idct, idctn, idst, idstn,
)
from scipy.fft import (
    dct as scipy_dct,
    dctn as scipy_dctn,
    dst as scipy_dst,
    dstn as scipy_dstn,
    idct as scipy_idct,
    idctn as scipy_idctn,
    idst as scipy_idst,
    idstn as scipy_idstn,
)
import torch
import pytest

devices = get_test_devices()
functions = ('dct', 'dst', 'idct', 'idst')
types = (1, 2, 3)
norms = ('forward', 'backward', 'ortho')


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("function", functions)
@pytest.mark.parametrize("type", types)
@pytest.mark.parametrize("norm", norms)
def test_dctdst(device, function, type, norm):
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
    scipy_function1 = (scipy_dct if function == 'dct' else
                       scipy_dst if function == 'dst' else
                       scipy_idct if function == 'idct' else
                       scipy_idst if function == 'idst' else None)
    scipy_functionn = (scipy_dctn if function == 'dct' else
                       scipy_dstn if function == 'dst' else
                       scipy_idctn if function == 'idct' else
                       scipy_idstn if function == 'idst' else None)

    def check(x, dim=-1):
        if not isinstance(dim, int):
            fn, scipy_fn = functionn, scipy_functionn
            scipy_kwargs = dict(axes=dim, type=type, norm=norm)
        else:
            fn, scipy_fn = function1, scipy_function1
            scipy_kwargs = dict(axis=dim, type=type, norm=norm)

        norm_fmath = norm
        if norm_fmath == 'ortho':
            norm_fmath += '_scipy'
        out_fmath = fn(x, dim=dim, type=type, norm=norm_fmath).cpu()
        out_scipy = torch.as_tensor(scipy_fn(x.cpu().numpy(), **scipy_kwargs))
        return torch.allclose(out_fmath, out_scipy)

    mat = torch.randn([4, 5], **backend)
    assert check(mat, dim=-1), "dim=-1"
    assert check(mat, dim=0), "dim=0"
    assert check(mat, dim=None), "dim=all"
    assert check(mat, dim=[0, 1]), "dim=[0, 1]"



from .utils import get_test_devices, init_device
from nitorch_fastmath.qr import eig_sym
import torch
import pytest

devices = get_test_devices()


@pytest.mark.parametrize("device", devices)
def test_symeig(device):
    device = init_device(device)
    dtype = torch.double
    backend = dict(dtype=dtype, device=device)

    def make_sym(mat):
        return (mat + mat.transpose(-1, -2)) / 2

    def check_symeig(mat):
        out_batch = eig_sym(mat).sort(-1)[0]
        out_nativ = torch.symeig(mat)[0].sort(-1)[0]
        return torch.allclose(out_batch, out_nativ)

    # 1x1
    mat = torch.randn([2, 1, 1], **backend)
    mat = make_sym(mat)
    assert check_symeig(mat), "1x1"

    # 2x2
    mat = torch.randn([2, 2, 2], **backend)
    assert check_symeig(mat), "2x2"

    # 3x3
    mat = torch.randn([2, 3, 3], **backend)
    assert check_symeig(mat), "3x3"

    # 4x4
    mat = torch.randn([2, 4, 4], **backend)
    assert check_symeig(mat), "4x4"


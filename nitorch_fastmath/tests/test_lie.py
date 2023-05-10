from .utils import get_test_devices, init_device
from nitorch_fastmath.lie import expm, logm
from scipy.linalg import expm as scipy_expm, logm as scipy_logm
import torch
import pytest

devices = get_test_devices()


@pytest.mark.parametrize("device", devices)
def test_expm(device):
    device = init_device(device)
    dtype = torch.double
    backend = dict(dtype=dtype, device=device)

    def check_expm(mat):
        out_batch = expm(mat).cpu()
        out_nativ = torch.as_tensor(scipy_expm(mat.cpu().numpy()))
        return torch.allclose(out_batch, out_nativ)

    # 1x1
    mat = torch.randn([2, 1, 1], **backend)
    assert check_expm(mat), "1x1"

    # 2x2
    mat = torch.randn([2, 2, 2], **backend)
    assert check_expm(mat), "2x2"

    # 3x3
    mat = torch.randn([2, 3, 3], **backend)
    assert check_expm(mat), "3x3"

    # 4x4
    mat = torch.randn([2, 4, 4], **backend)
    assert check_expm(mat), "4x4"


@pytest.mark.parametrize("device", devices)
def test_logm(device):
    # NOTE:
    # - currently, logm is only implemented for square matrices
    #   (because scipy's version is)
    # - our version always returns a real matrix if the input is real
    #   (even if the log is in reality complex)
    device = init_device(device)
    dtype = torch.double
    backend = dict(dtype=dtype, device=device)

    def check_logm(mat):
        out_batch = logm(mat).cpu()
        out_nativ = torch.as_tensor(scipy_logm(mat.cpu().numpy()))
        if out_nativ.is_complex():
            out_nativ = out_nativ.real
        return torch.allclose(out_batch, out_nativ)

    # 1x1
    mat = torch.randn([1, 1], **backend)
    assert check_logm(mat), "1x1"

    # 2x2
    mat = torch.randn([2, 2], **backend)
    assert check_logm(mat), "2x2"

    # 3x3
    mat = torch.randn([3, 3], **backend)
    assert check_logm(mat), "3x3"

    # 4x4
    mat = torch.randn([4, 4], **backend)
    assert check_logm(mat), "4x4"

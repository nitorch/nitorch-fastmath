from .utils import get_test_devices, init_device
from nitorch_fastmath.batched import batchmatvec, batchdet, batchinv
import torch
import pytest

devices = get_test_devices()


@pytest.mark.parametrize("device", devices)
def test_batchmatvec(device):
    device = init_device(device)

    def check_matvec(mat, vec):
        out_batch = batchmatvec(mat, vec)
        out_nativ = mat.matmul(vec.unsqueeze(-1)).squeeze(-1)
        return torch.allclose(out_batch, out_nativ)

    # 1x1
    mat = torch.randn([2, 1, 1], device=device)
    vec = torch.randn([2, 2, 1], device=device)
    assert check_matvec(mat, vec), "1x1"

    # 2x2
    mat = torch.randn([2, 2, 2], device=device)
    vec = torch.randn([2, 2, 2], device=device)
    assert check_matvec(mat, vec), "2x2"

    # 3x3
    mat = torch.randn([2, 3, 3], device=device)
    vec = torch.randn([2, 2, 3], device=device)
    assert check_matvec(mat, vec), "3x3"

    # 4x5
    mat = torch.randn([2, 4, 5], device=device)
    vec = torch.randn([2, 2, 5], device=device)
    assert check_matvec(mat, vec), "4x5"

    # mat longer
    mat = torch.randn([2, 2, 4, 5], device=device)
    vec = torch.randn([5], device=device)
    assert check_matvec(mat, vec), "mat longer"


@pytest.mark.parametrize("device", devices)
def test_batchdet(device):
    device = init_device(device)

    def check_det(mat):
        out_batch = batchdet(mat)
        out_nativ = torch.det(mat)
        return torch.allclose(out_batch, out_nativ)

    # 1x1
    mat = torch.randn([2, 1, 1], device=device)
    assert check_det(mat), "1x1"

    # 2x2
    mat = torch.randn([2, 2, 2], device=device)
    assert check_det(mat), "2x2"

    # 3x3
    mat = torch.randn([2, 3, 3], device=device)
    assert check_det(mat), "3x3"

    # 4x4
    mat = torch.randn([2, 4, 4], device=device)
    assert check_det(mat), "4x4"


@pytest.mark.parametrize("device", devices)
def test_batchinv(device):
    device = init_device(device)

    def check_inv(mat):
        out_batch = batchinv(mat)
        out_nativ = torch.linalg.inv(mat)
        return torch.allclose(out_batch, out_nativ)

    # 1x1
    mat = torch.randn([2, 1, 1], device=device)
    mat.diagonal(0, -1, -2).add_(10)
    assert check_inv(mat), "1x1"

    # 2x2
    mat = torch.randn([2, 2, 2], device=device)
    mat.diagonal(0, -1, -2).add_(10)
    assert check_inv(mat), "2x2"

    # 3x3
    mat = torch.randn([2, 3, 3], device=device)
    mat.diagonal(0, -1, -2).add_(10)
    assert check_inv(mat), "3x3"

    # 4x4
    mat = torch.randn([2, 4, 4], device=device)
    mat.diagonal(0, -1, -2).add_(10)
    assert check_inv(mat), "4x4"

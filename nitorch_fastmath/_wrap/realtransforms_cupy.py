from torch.utils.dlpack import to_dlpack, from_dlpack
import torch
from . import realtransforms_from_fft as F

try:
    from cupy import from_dlpack as cupy_from_dlpack, to_dlpack as cupy_to_dlpack
except ImportError:
    import cupy
    from cupy import fromDlpack as cupy_from_dlpack
    cupy_to_dlpack = cupy.ndarray.toDlpack


flipnorm = {'forward': 'backward', 'backward': 'forward', 'ortho': 'ortho'}


class DCTN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return from_cupy(F.dctn(to_cupy(x), type=type, axes=dim, norm=norm))

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = from_cupy(F.idctn(to_cupy(x), type=ctx.type, axes=ctx.dim, norm=norm))
        return x, None, None, None


class IDCTN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return from_cupy(F.idctn(to_cupy(x), type=type, axes=dim, norm=norm))

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = from_cupy(F.dctn(to_cupy(x), type=ctx.type, axes=ctx.dim, norm=norm))
        return x, None, None, None


class DSTN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return from_cupy(F.dstn(to_cupy(x), type=type, axes=dim, norm=norm))

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = from_cupy(F.idstn(to_cupy(x), type=ctx.type, axes=ctx.dim, norm=norm))
        return x, None, None, None


class IDSTN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        return from_cupy(F.idstn(to_cupy(x), type=type, axes=dim, norm=norm))

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = from_cupy(F.dstn(to_cupy(x), type=ctx.type, axes=ctx.dim, norm=norm))
        return x, None, None, None


def to_cupy(x):
    """Convert a torch tensor to cupy without copy"""
    return cupy_from_dlpack(to_dlpack(x))


def from_cupy(x):
    """Convert a cupy tensor to torch without copy"""
    return from_dlpack(cupy_to_dlpack(x))
import scipy.fft as F
import torch


flipnorm = {'forward': 'backward', 'backward': 'forward', 'ortho': 'ortho'}


class DCTN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        x = F.dctn(to_numpy(x), type=type, axes=dim, norm=norm)
        return from_numpy(x)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = F.idctn(to_numpy(x), type=ctx.type, axes=ctx.dim, norm=norm)
        return from_numpy(x), None, None, None


class IDCTN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        x = F.idctn(to_numpy(x), type=type, axes=dim, norm=norm)
        return from_numpy(x)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = F.dctn(to_numpy(x), type=ctx.type, axes=ctx.dim, norm=norm)
        return from_numpy(x), None, None, None


class DSTN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        x = F.dstn(to_numpy(x), type=type, axes=dim, norm=norm)
        return from_numpy(x)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = F.idstn(to_numpy(x), type=ctx.type, axes=ctx.dim, norm=norm)
        return from_numpy(x), None, None, None


class IDSTN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, type, dim, norm):
        ctx.dim = dim
        ctx.norm = norm or "backward"
        ctx.type = type
        x = F.idstn(to_numpy(x), type=type, axes=dim, norm=norm)
        return from_numpy(x)

    @staticmethod
    def backward(ctx, x):
        norm = flipnorm[ctx.norm]
        x = F.dstn(to_numpy(x), type=ctx.type, axes=ctx.dim, norm=norm)
        return from_numpy(x), None, None, None


def to_numpy(x):
    return x.detach().numpy()


def from_numpy(x):
    return torch.as_tensor(x)

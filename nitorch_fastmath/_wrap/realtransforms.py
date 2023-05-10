__all__ = [
    'dct', 'dst', 'idct', 'idst',
    'dctn', 'dstn', 'idctn', 'idstn',
]
import torch
from . import realtransforms_scipy as cpu
if torch.cuda.is_available():
    from . import realtransforms_cupy as cuda
else:
    cuda = None


def dct(x, dim=-1, norm=None, type=2):
    """Return the Discrete Cosine Transform

    !!! warning
        Type IV not implemented on the GPU

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    DCTN = cuda.DCTN if x.is_cuda else cpu.DCTN
    if type in (1, 2, 3, 4):
        return DCTN.apply(x, 1, dim, norm)
    else:
        raise ValueError('DCT only implemented for types I-IV')


def idct(x, dim=-1, norm=None, type=2):
    """Return the Inverse Discrete Cosine Transform

    !!! warning
        Type IV not implemented on the GPU

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    IDCTN = cuda.IDCTN if x.is_cuda else cpu.IDCTN
    if type in (1, 2, 3, 4):
        return IDCTN.apply(x, 1, dim, norm)
    else:
        raise ValueError('IDCT only implemented for types I-IV')


def dst(x, dim=-1, norm=None, type=2):
    """Return the Discrete Sine Transform

    !!! warning
        Type IV not implemented on the GPU

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    DSTN = cuda.DSTN if x.is_cuda else cpu.DSTN
    if type in (1, 2, 3, 4):
        return DSTN.apply(x, 1, dim, norm)
    else:
        raise ValueError('DST only implemented for types I-IV')


def idst(x, dim=-1, norm=None, type=2):
    """Return the Inverse Discrete Sine Transform

    !!! warning
        Type IV not implemented on the GPU

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : int
        Dimensions over which the DCT is computed.
        Default is the last one.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    IDSTN = cuda.IDSTN if x.is_cuda else cpu.IDSTN
    if type in (1, 2, 3, 4):
        return IDSTN.apply(x, 1, dim, norm)
    else:
        raise ValueError('DST only implemented for types I-IV')


def dctn(x, dim=None, norm=None, type=2):
    """Return multidimensional Discrete Cosine Transform
    along the specified axes.

    !!! warning
        Type IV not implemented on the GPU

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    DCTN = cuda.DCTN if x.is_cuda else cpu.DCTN
    if type in (1, 2, 3, 4):
        return DCTN.apply(x, 1, dim, norm)
    else:
        raise ValueError('DCT only implemented for types I-IV')


def idctn(x, dim=None, norm=None, type=2):
    """Return multidimensional Inverse Discrete Cosine Transform
    along the specified axes.

    !!! warning
        Type IV not implemented on the GPU

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    IDCTN = cuda.IDCTN if x.is_cuda else cpu.IDCTN
    if type in (1, 2, 3, 4):
        return IDCTN.apply(x, 1, dim, norm)
    else:
        raise ValueError('IDCT only implemented for types I-IV')


def dstn(x, dim=None, norm=None, type=2):
    """Return multidimensional Discrete Sine Transform
    along the specified axes.

    !!! warning
        Type IV not implemented on the GPU

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    DSTN = cuda.DSTN if x.is_cuda else cpu.DSTN
    if type in (1, 2, 3, 4):
        return DSTN.apply(x, 1, dim, norm)
    else:
        raise ValueError('DST only implemented for types I-IV')


def idstn(x, dim=None, norm=None, type=2):
    """Return multidimensional Inverse Discrete Sine Transform
    along the specified axes.

    !!! warning
        Type IV not implemented on the GPU

    Parameters
    ----------
    x : tensor
        The input tensor
    dim : [sequence of] int
        Dimensions over which the DCT is computed.
        If not given, all dimensions are used.
    norm : {“backward”, “ortho”, “forward”}
        Normalization mode. Default is “backward”.
    type: {1, 2, 3, 4}
        Type of the DCT. Default type is 2.

    Returns
    -------
    y : tensor
        The transformed tensor.

    """
    IDSTN = cuda.IDSTN if x.is_cuda else cpu.IDSTN
    if type in (1, 2, 3, 4):
        return IDSTN.apply(x, 1, dim, norm)
    else:
        raise ValueError('DST only implemented for types I-IV')

import torch
from types import GeneratorType as generator
import inspect
try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except ImportError:
    custom_fwd = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda x: x)
    custom_bwd = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda x: x)


def ensure_list(x, size=None, crop=True, **kwargs):
    """Ensure that an object is a list (of size at last dim)

    If x is a list, nothing is done (no copy triggered).
    If it is a tuple, it is converted into a list.
    Otherwise, it is placed inside a list.
    """
    if not isinstance(x, (list, tuple, range, generator)):
        x = [x]
    elif not isinstance(x, list):
        x = list(x)
    if size and len(x) < size:
        default = kwargs.get('default', x[-1])
        x += [default] * (size - len(x))
    if size and crop:
        x = x[:size]
    return x


def fast_slice_tensor(x, index, dim=-1):
    """Index a tensor along one dimensions.

    This function is relatively similar to `torch.index_select`, except
    that it uses the native indexing mechanism and can therefore
    returns a tensor that use the same storage as the input tensor.

    It is faster but less versatile than `slice_tensor`.

    Parameters
    ----------
    x : tensor
        Input tensor.
    index : int or list[int] or slice
        Indices to select along `dim`.
    dim : int, default=last
        Dimension to index.

    Returns
    -------
    y : tensor
        Output tensor.

    """
    slicer = [slice(None)] * x.dim()
    slicer[dim] = index
    slicer = tuple(slicer)
    return x[slicer]


def slice_tensor(x, index, dim=None):
    """Index a tensor along one or several dimensions.

    This function is relatively similar to `torch.index_select`, except
    that it uses the native indexing mechanism and can therefore
    returns a tensor that use the same storage as the input tensor.

    Parameters
    ----------
    x : tensor
        Input tensor.
    index : index_like or tuple[index_like]
        Indices to select along each dimension in `dim`.
        If multiple dimensions are indexed, they *must* be held in a
        tuple (not a list). Each index can be a long, list of long,
        slice or tensor of long, but *cannot* be an ellipsis or
        tensor of bool.
    dim : int or sequence[int], optional
        Dimensions to index. If it is a list, `index` *must* be a tuple.
        By default, the last `n` dimensions (where `n` is the number of
        indices in `index`) are used.


    Returns
    -------
    y : tensor
        Output tensor.

    """
    # format (dim, index) as (list, tuple) with same length
    if not isinstance(index, tuple):
        index = (index,)
    if dim is None:
        dim = list(range(-len(index), 0))
    dim = ensure_list(dim)
    nb_dim = max(len(index), len(dim))
    dim = ensure_list(dim, nb_dim)
    index = tuple(ensure_list(index, nb_dim))

    # build index
    full_index = [slice(None)] * x.dim()
    for d, ind in zip(dim, index):
        if ind is Ellipsis or (torch.is_tensor(ind) and
                               ind.dtype == torch.bool):
            raise TypeError('`index` cannot be an ellipsis or mask')
        full_index[d] = ind
    full_index = tuple(full_index)

    return x.__getitem__(full_index)


def cumprod(sequence, reverse=False, exclusive=False):
    """Perform the cumulative product of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__mul__` operator is defined.
    reverse : bool, default=False
        Compute cumulative product from right-to-left:
        `cumprod([a, b, c], reverse=True) -> [a*b*c, b*c, c]`
    exclusive : bool, default=False
        Exclude self from the cumulative product:
        `cumprod([a, b, c], exclusive=True) -> [1, a, a*b]`

    Returns
    -------
    product : list
        Product of the elements in the sequence.

    """
    if reverse:
        sequence = reversed(sequence)
    accumulate = None
    seq = [1] if exclusive else []
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        else:
            accumulate = accumulate * elem
        seq.append(accumulate)
    if exclusive:
        seq = seq[:-1]
    if reverse:
        seq = list(reversed(seq))
    return seq


def sub2ind(subs, shape, out=None):
    """Convert sub indices (i, j, k) into linear indices.

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    subs : (D, ...) tensor_like
        List of sub-indices. The first dimension is the number of dimension.
        Each element should have the same number of elements and shape.
    shape : (D,) vector_like
        Size of each dimension. Its length should be the same as the
        first dimension of ``subs``.
    out : tensor, optional
        Output placeholder

    Returns
    -------
    ind : (...) tensor
        Linear indices
    """
    *subs, ind = subs
    if out is None:
        ind = torch.as_tensor(ind).clone()
    else:
        out.reshape(ind.shape).copy_(ind)
        ind = out
    bck = dict(dtype=ind.dtype, device=ind.device)
    stride = cumprod(shape[1:], reverse=True)
    for i, s in zip(subs, stride):
        ind += torch.as_tensor(i, **bck) * torch.as_tensor(s, **bck)
    return ind


# floor_divide returns wrong results for negative values, because it truncates
# instead of performing a proper floor. In recent version of pytorch, it is
# advised to use div(..., rounding_mode='trunc'|'floor') instead.
# Here, we only use floor_divide on positive values so we do not care.
try:
    _one = torch.as_tensor(1.)
    torch.div(_one, _one, rounding_mode='trunc')
    def _trunc_div(*a, **k):
        return torch.div(*a, **k, rounding_mode='trunc')
except Exception:
    _trunc_div = torch.floor_divide


def ind2sub(ind, shape, out=None):
    """Convert linear indices into sub indices (i, j, k).

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    ind : tensor_like
        Linear indices
    shape : (D,) vector_like
        Size of each dimension.
    out : tensor, optional
        Output placeholder

    Returns
    -------
    subs : (D, ...) tensor
        Sub-indices.
    """
    ind = torch.as_tensor(ind)
    bck = dict(dtype=ind.dtype, device=ind.device)
    stride = cumprod(shape, reverse=True, exclusive=True)
    stride = torch.as_tensor(stride, **bck)
    if out is None:
        sub = ind.new_empty([len(shape), *ind.shape])
    else:
        sub = out.reshape([len(shape), *ind.shape])
    sub[:, ...] = ind
    for d in range(len(shape)):
        if d > 0:
            torch.remainder(sub[d], torch.as_tensor(stride[d-1], **bck), out=sub[d])
        sub[d] = _trunc_div(sub[d], stride[d], out=sub[d])
    return sub


def eps(dtype='float32'):
    """Machine epsilon for different precisions."""
    f16_types = []
    if hasattr(torch, 'float16'):
        f16_types += ['float16', torch.float16]
    if hasattr(torch, 'complex32'):
        f16_types += ['complex32', torch.complex32]
    f32_types = ['float32', torch.float32, 'complex64', torch.complex64]
    f64_types = ['float64', torch.float64, 'complex128', torch.complex128]

    if dtype in f16_types:
        return 2 ** -10
    if dtype in f32_types:
        return 2 ** -23
    elif dtype in f64_types:
        return 2 ** -52
    else:
        raise NotImplementedError

r'''
## Overview

This first section reimplements several reduction functions (sum, mean,
max...), with a more consistent API than the native pytorch one:
- all functions can reduce across multiple dimensions simultaneously
- min/max/median functions only return the reduced tensor by default
  (not the original indices of the returned elements).
  They have a specific argument `return_indices` to request these indices.
- all functions have an `omitnan` argument, or alternatively a `nan`
  version (e.g., `nansum`) where `omitnan=True` by default.
The typical API for all functions is:

```python
def fn(input, dim=None, keepdim=False, omitnan=False, inplace=False, out=None): ...
  """
  input   : tensor, Input tensor
  dim     : int or sequence[int], Dimensions to reduce (default: all)
  keepdim : bool, Do not sequeeze reduced dimensions (default: False)
  omitnan : bool, Discard NaNs form the reduction (default: False)
  inplace : bool, Allow the input tensor to be modified (default: False)
                  (only useful in conjunction with `omitnan=True`)
  out     : tensor, Output placeholder
  """
```

Reduction functions that pick a value from the input tensor (e.g., `max`)
have the additional argument:

```python
def fn(..., return_indices=False): ...
  """
  return_indices : bool, Also return indices of the picked elements
  """
```
'''
__all__ = [
    'min', 'max', 'nanmin', 'nanmax', 'median',
    'sum', 'nansum', 'mean', 'nanmean', 'var', 'nanvar', 'std', 'nanstd'
]
import torch
from torch import Tensor
from typing import Optional, TypeVar, Tuple, Union, Sequence
from .typing import OneOrTwo, OneOrSeveral
from .utils import ind2sub, ensure_list


def _reduce_index(fn, input, dim=None, keepdim=False, omitnan=False,
                  inplace=False, return_indices=False, out=None,
                  nanfn=lambda x: x):
    """Multi-dimensional reduction for min/max/median.

    Signatures
    ----------
    fn(input) -> Tensor
    fn(input, dim) -> Tensor
    fn(input, dim, return_indices=True) -> (Tensor, Tensor)

    Parameters
    ----------
    fn : callable
        Reduction function
    input : tensor
        Input tensor.
    dim : int or sequence[int]
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    omitnan : bool, default=False
        Omit NaNs (else, the reduction of a NaN and any other value is NaN)
    inplace : bool, defualt=False
        Allow modifying the input tensor in-place
        (Only useful if `omitnan == True`)
    return_indices : bool, defualt=False
        Return index of the min/max value on top of the value
    out : tensor or (tensor, tensor), optional
        Output placeholder
    nanfn : callable, optional
        Preprocessing function for removing nans

    Returns
    -------
    output : tensor
        Reduced tensor
    indices : (..., [len(dim)]) tensor
        Indices of the min/max/median values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    if omitnan:
        # If omitnan, we call a function that does the pre and post processing
        # of nans and give it a pointer to ourselves so that it can call us
        # back
        def self(inp):
            return _reduce_index(fn, inp, dim=dim, keepdim=keepdim, omitnan=False,
                                 inplace=False, return_indices=return_indices,
                                 out=out)
        return nanfn(self, input, inplace)

    input = torch.as_tensor(input)
    if dim is None:
        # If min across the entire tensor -> call torch
        return fn(input)

    # compute shapes
    scalar_dim = torch.as_tensor(dim).dim() == 0
    dim = [d if d >= 0 else input.dim() + d for d in ensure_list(dim)]
    shape = input.shape
    subshape = [s for d, s in enumerate(shape) if d not in dim]
    keptshape = [s if d not in dim else 1 for d, s in enumerate(shape)]
    redshape = [shape[d] for d in dim]
    input = torch.movedim(input, dim, -1)    # move reduced dim to the end
    input = input.reshape([*subshape, -1])   # collapse reduced dimensions

    # prepare placeholder
    out_val, out_ind = ensure_list(out, 2, default=None)
    if out_ind and len(dim) > 1:
        out_ind_tmp = input.new_empty(subshape, dtype=torch.long)
    else:
        out_ind_tmp = out_ind
    if out_val is not None and out_ind_tmp is None:
        out_ind_tmp = input.new_empty(subshape, dtype=torch.long)
    elif out_ind_tmp is not None and out_val is None:
        out_val = input.new_empty(subshape)
    out = (out_val, out_ind_tmp) if out_val is not None else None

    input, indices = fn(input, dim=-1, out=out)       # perform reduction

    if keepdim:
        input = input.reshape(keptshape)  # keep reduced singleton dimensions

    if return_indices:
        # convert to (i, j, k) indices
        indices = ind2sub(indices, redshape, out=out_ind)
        indices = torch.movedim(indices, 0, -1)
        if keepdim:
            indices = indices.reshape([*keptshape, -1])
        if scalar_dim:
            indices = indices[..., 0]
        return input, indices

    return input


def max(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    omitnan: bool = False,
    inplace: bool = False,
    return_indices: bool = False,
    out: Optional[OneOrTwo[Tensor]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Multi-dimensional max reduction.

    Signatures
    ----------
    ```python
    max(input) -> Tensor
    max(input, dim) -> Tensor
    max(input, dim, return_indices=True) -> (Tensor, Tensor)
    ```

    Notes
    -----
    - This function cannot compute the maximum of two tensors, it only
       computes the maximum of one tensor (along a dimension).

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or sequence[int]`
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    omitnan : `bool`, default=False
        Omit NaNs (else, the reduction of a NaN and any other value is NaN)
    inplace : `bool`, default=False
        Allow modifying the input tensor in-place
        (Only useful if `omitnan == True`)
    return_indices : `bool`, default=False
        Return index of the max value on top of the value
    out : `tensor or (tensor, tensor),` optional
        Output placeholder

    Returns
    -------
    output : `tensor`
        Reduced tensor
    indices : `(..., [len(dim)]) tensor`
        Indices of the max values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    opt = dict(dim=dim, keepdim=keepdim, omitnan=omitnan, inplace=inplace,
               return_indices=return_indices, out=out)
    return _reduce_index(torch.max, input, **opt, nanfn=_nanmax)


def min(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    omitnan: bool = False,
    inplace: bool = False,
    return_indices: bool = False,
    out: Optional[OneOrTwo[Tensor]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Multi-dimensional min reduction.

    Signatures
    ----------
    ```python
    min(input) -> Tensor
    min(input, dim) -> Tensor
    min(input, dim, return_indices=True) -> (Tensor, Tensor)
    ```

    Notes
    -----
    - This function cannot compute the minimum of two tensors, it only
       computes the minimum of one tensor (along a dimension).

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or sequence[int]`
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    omitnan : `bool`, default=False
        Omit NaNs (else, the reduction of a NaN and any other value is NaN)
    inplace : `bool`, default=False
        Allow modifying the input tensor in-place
        (Only useful if `omitnan == True`)
    return_indices : `bool`, default=False
        Return index of the min value on top of the value
    out : `tensor or (tensor, tensor)`, optional
        Output placeholder

    Returns
    -------
    output : `tensor`
        Reduced tensor
    indices : `(..., [len(dim)]) tensor`
        Indices of the min values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    opt = dict(dim=dim, keepdim=keepdim, omitnan=omitnan, inplace=inplace,
               return_indices=return_indices, out=out)
    return _reduce_index(torch.min, input, **opt, nanfn=_nanmin)


def _nanmax(fn, input, inplace=False):
    """Replace `nan`` with `-inf`"""
    input = torch.as_tensor(input)
    mask = torch.isnan(input)
    if inplace and not input.requires_grad:
        input[mask] = -float('inf')
    else:
        val_ninf = input.new_full([], -float('inf'))
        input = torch.where(mask, val_ninf, input)
    return fn(input)


def nanmax(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    inplace: bool = False,
    return_indices: bool = False,
    out: Optional[OneOrTwo[Tensor]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Multi-dimensional max reduction, excluding NaNs.

    Signatures
    ----------
    ```python
    nanmax(input) -> Tensor
    nanmax(input, dim) -> Tensor
    nanmax(input, dim, return_indices=True) -> (Tensor, Tensor)
    ```

    Notes
    -----
    - This function cannot compute the minimum of two tensors, it only
       computes the minimum of one tensor (along a dimension).

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or sequence[int]`
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    inplace : `bool`, default=False
        Allow modifying the input tensor in-place
        (Only useful if `omitnan == True`)
    return_indices : `bool`, default=False
        Return index of the max value on top of the value
    out : `tensor or (tensor, tensor)`, optional
        Output placeholder

    Returns
    -------
    output : `tensor`
        Reduced tensor
    indices : `(..., [len(dim)]) tensor`
        Indices of the max values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    opt = dict(dim=dim, keepdim=keepdim, inplace=inplace,
               return_indices=return_indices, out=out)
    return max(input, **opt, omitnan=True)


def _nanmin(fn, input, inplace=False):
    """Replace `nan`` with `inf`"""
    input = torch.as_tensor(input)
    mask = torch.isnan(input)
    if inplace and not input.requires_grad:
        input[mask] = float('inf')
    else:
        val_inf = input.new_full([], float('inf'))
        input = torch.where(mask, val_inf, input)
    return fn(input)


def nanmin(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    inplace: bool = False,
    return_indices: bool = False,
    out: Optional[OneOrTwo[Tensor]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Multi-dimensional min reduction, excluding NaNs.

    Signatures
    ----------
    ```python
    nanmin(input) -> Tensor
    nanmin(input, dim) -> Tensor
    nanmin(input, dim, return_indices=True) -> (Tensor, Tensor)
    ```

    Notes
    -----
    - This function cannot compute the minimum of two tensors, it only
       computes the minimum of one tensor (along a dimension).

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or sequence[int]`
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    inplace : `bool`, default=False
        Allow modifying the input tensor in-place
        (Only useful if `omitnan == True`)
    return_indices : `bool`, default=False
        Return index of the min value on top of the value
    out : `tensor or (tensor, tensor`), optional
        Output placeholder

    Returns
    -------
    output : `tensor`
        Reduced tensor
    indices : `(..., [len(dim)]) tensor`
        Indices of the min values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    opt = dict(dim=dim, keepdim=keepdim, inplace=inplace,
               return_indices=return_indices, out=out)
    return min(input, **opt, omitnan=True)


def median(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    omitnan: bool = False,
    inplace: bool = False,
    return_indices: bool = False,
    out: Optional[OneOrTwo[Tensor]] = None,
) -> Tensor:
    """Multi-dimensional median reduction.

    Signatures
    ----------
    ```python
    median(input) -> Tensor
    median(input, dim) -> Tensor
    median(input, dim, return_indices=True) -> (Tensor, Tensor)
    ```

    Note
    ----
    - This function always omits NaNs

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or sequence[int]`
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    return_indices : `bool`, default=False
        Return index of the median value on top of the value
    out : `tensor or (tensor, tensor)`, optional
        Output placeholder

    Returns
    -------
    output : `tensor`
        Reduced tensor
    indices : `(..., [len(dim)]) tensor`
        Indices of the median values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    opt = dict(dim=dim, keepdim=keepdim, return_indices=return_indices, out=out)
    return _reduce_index(torch.median, input, **opt)


def sum(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    omitnan: bool = False,
    inplace: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None
) -> Tensor:
    """Compute the sum of a tensor.

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or list[int]`, optional
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    omitnan : `bool`, default=False
        Omit NaNs in the sum.
    inplace : `bool`, default=False
        Authorize working inplace.
    dtype : `dtype`, default=input.dtype
        Accumulator data type
    out : `tensor`, optional
        Output placeholder.

    Returns
    -------
    out : `tensor`
        Output tensor
    """
    kwargs = dict(dim=dim, keepdim=keepdim, dtype=dtype, out=out)
    if omitnan:
        return nansum(input, inplace=inplace, **kwargs)
    else:
        return torch.sum(input, **kwargs)


def nansum(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    inplace: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None
) -> Tensor:
    """Compute the sum of a tensor, excluding nans.

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or list[int]`, optional
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    inplace : `bool`, default=False
        Authorize working inplace.
    dtype : `dtype`, default=input.dtype
        Accumulator data type
    out : `tensor`, optional
        Output placeholder.

    Returns
    -------
    out : tensor
        Output tensor
    """
    kwargs = dict(dim=dim, keepdim=keepdim, dtype=dtype, out=out)
    if not inplace:
        input = input.clone()
    mask = torch.isnan(input)
    if input.requires_grad and input.is_leaf:
        zero = torch.as_tensor(0, dtype=input.dtype, device=input.device)
        input = torch.where(mask, zero, input)
    else:
        input.masked_fill_(mask, 0)
    return torch.sum(input, **kwargs)


def mean(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    omitnan: bool = False,
    inplace: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None
) -> Tensor:
    """Compute the mean of a tensor.

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or list[int]`, optional
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    omitnan : `bool`, default=False
        Omit NaNs in the sum.
    inplace : `bool`, default=False
        Authorize working inplace.
    dtype : `dtype`, default=input.dtype
        Accumulator data type
    out : `tensor`, optional
        Output placeholder.

    Returns
    -------
    out : tensor
        Output tensor
    """
    kwargs = dict(dim=dim, keepdim=keepdim, dtype=dtype, out=out)
    if omitnan:
        return nanmean(input, inplace=inplace, **kwargs)
    else:
        return torch.mean(input, **kwargs)


def nanmean(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    inplace: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None
) -> Tensor:
    """Compute the mean of a tensor, excluding nans.

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or list[int]`, optional
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    inplace : `bool`, default=False
        Authorize working inplace.
    dtype : `dtype`, default=input.dtype
        Accumulator data type
    out : `tensor`, optional
        Output placeholder.

    Returns
    -------
    out : `tensor`
        Output tensor
    """
    kwargs = dict(dim=dim, keepdim=keepdim, dtype=dtype, out=out)
    if not inplace:
        input = input.clone()
    mask = torch.isnan(input)
    if input.requires_grad and input.is_leaf:
        zero = torch.as_tensor(0, dtype=input.dtype, device=input.device)
        input = torch.where(mask, zero, input)
    else:
        input.masked_fill_(mask, 0)
    mask = mask.bitwise_not_()
    weights = mask.sum(**kwargs).to(kwargs.get('dtype', input.dtype))
    return torch.sum(input, **kwargs) / weights


def var(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    unbiased: bool = True,
    omitnan: bool = False,
    inplace: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None
) -> Tensor:
    """Compute the variance of a tensor, excluding nans.

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or list[int]`, optional
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    unbiased : `bool`, default=True
        Whether to use the unbiased estimation or not.
    omitnan : `bool`, default=False
        Omit NaNs.
    inplace : `bool`, default=False
        Authorize working inplace.
    dtype : `dtype`, default=input.dtype
        Accumulator data type

    Returns
    -------
    out : `tensor`
        Output tensor
    """
    kwargs = dict(dim=dim, keepdim=keepdim, unbiased=unbiased, dtype=dtype, out=out)
    if omitnan:
        return nanvar(input, inplace=inplace, **kwargs)
    else:
        return torch.var(input, **kwargs)


def nanvar(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    unbiased: bool = True,
    inplace: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None
) -> Tensor:
    """Compute the variance of a tensor, excluding nans.

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or list[int]`, optional
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    unbiased : `bool`, default=True
        Whether to use the unbiased estimation or not.
    inplace : `bool`, default=False
        Authorize working inplace.
    dtype : `dtype`, default=input.dtype
        Accumulator data type

    Returns
    -------
    out : `tensor`
        Output tensor
    """
    kwargs = dict(dim=dim, keepdim=keepdim, dtype=dtype, out=out)
    requires_grad = input.requires_grad
    inplace = inplace and not (requires_grad and input.is_leaf)
    if not inplace:
        input = input.clone()
    mask = torch.isnan(input)
    input.masked_fill_(mask, 0)
    mask = mask.bitwise_not_()
    weights = mask.sum(**kwargs).to(kwargs.get('dtype', input.dtype))
    mean = torch.sum(input, **kwargs).div_(weights)
    input = input.square() if requires_grad else input.square_()
    var = torch.sum(input, **kwargs).div_(weights)
    var -= mean
    if unbiased:
        weights /= (weights - 1)
        var *= weights
    return var


def std(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    unbiased: bool = True,
    omitnan: bool = False,
    inplace: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None
) -> Tensor:
    """Compute the standard deviation of a tensor, excluding nans.

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or list[int]`, optional
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    unbiased : `bool`, default=True
        Whether to use the unbiased estimation or not.
    omitnan : `bool`, default=False
        Omit NaNs.
    inplace : `bool`, default=False
        Authorize working inplace.
    dtype : `dtype`, default=input.dtype
        Accumulator data type

    Returns
    -------
    out : `tensor`
        Output tensor
    """
    kwargs = dict(dim=dim, keepdim=keepdim, unbiased=unbiased, dtype=dtype, out=out)
    if omitnan:
        return nanstd(input, inplace=inplace, **kwargs)
    else:
        return torch.std(input, **kwargs)


def nanstd(
    input: Tensor,
    dim: Optional[OneOrSeveral[int]] = None,
    keepdim: bool = False,
    unbiased: bool = True,
    inplace: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None
) -> Tensor:
    """Compute the standard deviation of a tensor, excluding nans.

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int or list[int]`, optional
        Dimensions to reduce.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    unbiased : `bool`, default=True
        Whether to use the unbiased estimation or not.
    inplace : `bool`, default=False
        Authorize working inplace.
    dtype : `dtype`, default=input.dtype
        Accumulator data type

    Returns
    -------
    out : `tensor`
        Output tensor
    """
    kwargs = dict(dim=dim, keepdim=keepdim, dtype=dtype, out=out)
    input = nanvar(input, inplace=inplace, unbiased=unbiased, **kwargs)
    input = input.sqrt_() if not input.requires_grad else input.sqrt()
    return input

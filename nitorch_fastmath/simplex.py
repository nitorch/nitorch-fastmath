"""
## Overview

This module is concerned with functions that deal with data lying on the simplex,
_i.e._, probabilities.
Specifically, we implement `softmax`, `log_softmax`, `logsumexp` and `logit`.

While most of these functions already exist in  PyTorch, we define more
generic function that accept an "implicit" class.

This implicit class exists due to the constrained nature of discrete
probabilities, which must sum to one, meaning that their space ("the simplex")
has one less dimensions than the number of classes.

Similarly, we can restrain the logit (= log probability) space to be of
dimension K-1 by forcing one of the classes to have logits of arbitrary value
(e.g., zero). This trick makes functions like softmax invertible.

Note that in the 2-class case, it is extremely common to work in this
implicit setting by using the sigmoid function over a single logit instead
of the softmax function over two logits.

All functions below accept an argument `implicit` which takes either one
(boolean) value or a tuple of two (boolean) values. The first value
specifies if the input tensor has an explicit class while the second value
specified if the output tensor should have an implicit class.

Note that to minimize the memory footprint and numerical errors, most
backward passes are explicitly reimplemented (rather than relying on
automatic differentiation). This is because these function involve multiple
calls to `log` and `exp`, which must all store their input in order to
backpropagate, whereas a single tensor needs to be stored to backpropagate
through the entire softmax function.

---
"""
__all__ = [
    'logsumexp',
    'softmax',
    'log_softmax',
    'logit',
    'softmax_lse',
]
import torch
from torch import Tensor
from typing import Tuple, Optional
from .typing import OneOrTwo
from .utils import slice_tensor, ensure_list, custom_fwd, custom_bwd


def logsumexp(
        input: Tensor,
        dim: int = -1,
        keepdim: bool = False,
        implicit: bool = False,
) -> Tensor:
    """Numerically stabilised log-sum-exp (lse).

    Parameters
    ----------
    input : `tensor`
        Input tensor.
    dim : `int`, default=-1
        The dimension or dimensions to reduce.
    keepdim : `bool`, default=False
        Whether the output tensor has dim retained or not.
    implicit : `bool`, default=False
        Assume that an additional (hidden) channel with value zero exists.

    Returns
    -------
    lse : `tensor`
        Output tensor.
    """
    return _LSE.apply(input, dim, keepdim, implicit)


def _lse_fwd(input, dim=-1, keepdim=False, implicit=False):
    input = torch.as_tensor(input).clone()

    lse = input.max(dim=dim, keepdim=True)[0]
    if implicit:
        zero = input.new_zeros([])
        lse = torch.max(lse, zero)

    input = input.sub_(lse).exp_().sum(dim=dim, keepdim=True)
    if implicit:
        input += lse.neg().exp_()
    lse += input.log_()

    if not keepdim:
        lse = lse.squeeze(dim=dim)

    return lse


def _lse_bwd(input, output_grad, dim=-1, keepdim=False, implicit=False):
    input = _softmax_fwd(input, dim, implicit)
    if not keepdim:
        output_grad = output_grad.unsqueeze(dim)
    input *= output_grad
    return input


class _LSE(torch.autograd.Function):
    """Log-Sum-Exp with implicit class."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input, dim, keepdim, implicit):

        # Save precomputed components of the backward pass
        needs_grad = torch.is_tensor(input) and input.requires_grad
        if needs_grad:
            ctx.save_for_backward(input)
            ctx.args = {'dim': dim, 'implicit': implicit, 'keepdim': keepdim}

        return _lse_fwd(input, dim=dim, keepdim=keepdim, implicit=implicit)

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):

        input, = ctx.saved_tensors
        return _lse_bwd(input, output_grad,
                        dim=ctx.args['dim'],
                        keepdim=ctx.args['keepdim'],
                        implicit=ctx.args['implicit']), None, None, None


def _add_class(x, bg, dim, index):
    # for implicit softmax
    if isinstance(bg, (int, float)):
        bg = torch.as_tensor(bg, dtype=x.dtype, device=x.device)
        bgshape = list(x.shape)
        bgshape[dim] = 1
        bg = bg.expand(bgshape)
    if index in (-1, x.shape[dim]-1):
        pieces = [x, bg]
    elif index in (0, -dim):
        pieces = [bg, x]
    else:
        pieces = [
            slice_tensor(x, slice(index), dim),
            bg,
            slice_tensor(x, slice(index, None), dim)]
    return torch.cat(pieces, dim=dim)


def _remove_class(x, dim, index):
    # for implicit softmax
    if index in (-1, x.shape[dim]-1):
        x = slice_tensor(x, slice(-1), dim)
    elif index in (0, -dim):
        x = slice_tensor(x, slice(1, None), dim)
    else:
        x = torch.cat([
            slice_tensor(x, slice(index), dim),
            slice_tensor(x, slice(index+1, None), dim)])
    return x


def softmax(
        input: Tensor,
        dim: int = -1,
        implicit: OneOrTwo[bool] = False,
        implicit_index: int = 0,
) -> Tensor:
    """ SoftMax (safe).

    Parameters
    ----------
    input : `tensor`
        Tensor with values.
    dim : `int`, default=-1
        Dimension to take softmax, defaults to last dimensions.
    implicit : `bool or (bool, bool)`, default=False
        The first value relates to the input tensor and the second
        relates to the output tensor.

        - `implicit[0] == True` assumes that an additional (hidden) channel
          with value zero exists.
        - `implicit[1] == True` drops the last class from the
          softmaxed tensor.
    implicit_index : `int`, default=0
        Index of the implicit class.

    Returns
    -------
    output : `tensor`
        Soft-maxed tensor with values.
    """
    input = torch.as_tensor(input)
    return _Softmax.apply(input, dim, implicit, implicit_index)


def _softmax_fwd(input, dim=-1, implicit=False, implicit_index=0):
    implicit_in, implicit_out = ensure_list(implicit, 2)

    maxval, _ = torch.max(input, dim=dim, keepdim=True)
    if implicit_in:
        maxval.clamp_min_(0)  # don't forget the class full of zeros

    input = input.clone().sub_(maxval).exp_()
    sumval = torch.sum(input, dim=dim, keepdim=True,
                       out=maxval if not implicit_in else None)
    if implicit_in:
        sumval += maxval.neg().exp_()  # don't forget the class full of zeros
    input /= sumval

    if implicit_in and not implicit_out:
        background = input.sum(dim, keepdim=True).neg_().add_(1)
        input = _add_class(input, background, dim, implicit_index)
    elif implicit_out and not implicit_in:
        input = _remove_class(input, dim, implicit_index)

    return input


def _softmax_bwd(output, output_grad, dim=-1, implicit=False, implicit_index=0):
    implicit = ensure_list(implicit, 2)
    add_dim = implicit[1] and not implicit[0]
    drop_dim = implicit[0] and not implicit[1]

    grad = output_grad.clone()
    del output_grad
    grad *= output
    gradsum = grad.sum(dim=dim, keepdim=True)
    grad = grad.addcmul_(gradsum, output, value=-1)  # grad -= gradsum * output
    if add_dim:
        grad_background = output.sum(dim=dim, keepdim=True).neg().add(1)
        grad_background.mul_(gradsum.neg_())
        grad = _add_class(grad, grad_background, dim, implicit_index)
    elif drop_dim:
        grad = slice_tensor(grad, slice(-1), dim)

    return grad


class _Softmax(torch.autograd.Function):
    """Softmax with implicit class."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input, dim, implicit, implicit_index):

        # Save precomputed components of the backward pass
        needs_grad = torch.is_tensor(input) and input.requires_grad
        # Compute matrix exponential
        s = _softmax_fwd(input, dim=dim, implicit=implicit,
                         implicit_index=implicit_index)

        if needs_grad:
            ctx.save_for_backward(s)
            ctx.args = {'dim': dim, 'implicit': implicit}

        return s

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):

        s, = ctx.saved_tensors
        return _softmax_bwd(s, output_grad,  dim=ctx.args['dim'],
                            implicit=ctx.args['implicit']), None, None, None


def logit(
        input: Tensor,
        dim: int = -1,
        implicit: OneOrTwo[bool] = False,
        implicit_index: int = 0,
) -> Tensor:
    """(Multiclass) logit function

    Notes
    -----
    - $\operatorname{logit}(\mathbf{x})_k = \log(x_k) - \log(x_K)$,
      where K is an arbitrary channel.
    - The `logit` function is the inverse of the `softmax` function:
        - `logit(softmax(x, implicit=True), implicit=True) == x`
        - `softmax(logit(x, implicit=True), implicit=True) == x`
    - Note that when `implicit=False`, `softmax` is surjective (many
       possible logits map to the same simplex value). We only have:
        - `softmax(logit(x, implicit=False), implicit=False) == x`
    - `logit(x, implicit=True)`, with `x.shape[dim] == 1` is equivalent
       to the "classical" binary logit function (inverse of the sigmoid).

    Parameters
    ----------
    input : `tensor`
        Tensor of probabilities.
    dim : `int`, default=-1
        Simplex dimension, along which the logit is performed.
    implicit : `bool or (bool, bool)`, default=False
        The first value relates to the input tensor and the second
        relates to the output tensor.

        - `implicit[0] == True` assumes that an additional (hidden) channel
          exists, such as the sum along `dim` is one.
        - `implicit[1] == True` drops the implicit channel from the
          logit tensor.
    implicit_index : `int`, default=0
        Index of the implicit channel. This is the channel whose logits
        are assumed equal to zero.

    Returns
    -------
    output : `tensor`
    """
    implicit = ensure_list(implicit, 2)
    if implicit[0]:
        input_extra = input.sum(dim).neg_().add_(1).clamp_min_(1e-8).log_()
        input = input.log()
    else:
        input = input.log()
        input_extra = slice_tensor(input, implicit_index, dim)
        if implicit[1]:
            input = _remove_class(input, dim, implicit_index)
    input_extra = input_extra.unsqueeze(dim)
    input -= input_extra.clone()
    if implicit[0] and not implicit[1]:
        input = _add_class(input, 0, dim, implicit_index)
    return input


def log_softmax(
        input: Tensor,
        dim: int = -1,
        implicit: OneOrTwo[bool] = False,
        implicit_index: int = 0,
) -> Tensor:
    """ Log(SoftMax).

    Parameters
    ----------
    input : `tensor`
        Tensor with values.
    dim : `int`, default=-1
        Dimension to take softmax, defaults to last dimensions.
    implicit : `bool or (bool, bool)`, default=False
        The first value relates to the input tensor and the second
        relates to the output tensor.

        - `implicit[0] == True` assumes that an additional (hidden) channel
          with value zero exists.
        - `implicit[1] == True` drops the last class from the
          softmaxed tensor.
    implicit_index : `int`, default=0

    Returns
    -------
    output : `tensor`
        Log-Softmaxed tensor with values.
    """
    input = torch.as_tensor(input)
    implicit = ensure_list(implicit, 2)
    lse = logsumexp(input, dim=dim, implicit=implicit[0], keepdim=True)
    if implicit[0] and not implicit[1]:
        output = _add_class(input, 0, dim, implicit_index)
        output -= lse
    elif implicit[1] and not implicit[0]:
        input = _remove_class(input, dim, implicit_index)
        output = input - lse
    else:
        output = input - lse
    return output


def softmax_lse(
        input: Tensor,
        dim: int = -1,
        weights: Optional[Tensor] = None,
        implicit: OneOrTwo[bool] = False,
) -> Tuple[Tensor, Tensor]:
    """ SoftMax (safe).

    Parameters
    ----------
    input : `tensor`
        Tensor with values.
    dim : `int`, default=-1
        Dimension to take softmax, defaults to last dimensions.
    weights : `tensor`, optional:
        Observation weights (only used in the log-sum-exp).
    implicit : `bool or (bool, bool)`, default=False
        The first value relates to the input tensor and the second
        relates to the output tensor.

        - `implicit[0] == True` assumes that an additional (hidden) channel
          with value zero exists.
        - `implicit[1] == True` drops the last class from the
          softmaxed tensor.

    Returns
    -------
    softmax : `tensor`
        Softmaxed tensor with values.
    lse : `tensor`
        Logsumexp
    """
    implicit_in, implicit_out = ensure_list(implicit, 2)

    maxval, _ = torch.max(input, dim=dim, keepdim=True)
    if implicit_in:
        maxval.clamp_min_(0)  # don't forget the class full of zeros

    input = (input-maxval).exp()
    sumval = torch.sum(input, dim=dim, keepdim=True)
    if implicit_in:
        sumval += maxval.neg().exp()  # don't forget the class full of zeros
    input = input / sumval

    # Compute log-sum-exp
    #   maxval = max(logit)
    #   lse = maxval + log[sum(exp(logit - maxval))]
    # If implicit
    #   maxval = max(max(logit),0)
    #   lse = maxval + log[sum(exp(logit - maxval)) + exp(-maxval)]
    sumval = sumval.log()
    maxval += sumval
    if weights is not None:
        maxval *= weights
    maxval = maxval.sum(dtype=torch.float64)

    if implicit_in and not implicit_out:
        background = input.sum(dim, keepdim=True).neg().add(1)
        input = torch.cat((input, background), dim=dim)
    elif implicit_out and not implicit_in:
        input = slice_tensor(input, slice(-1), dim)

    return input, maxval


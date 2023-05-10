"""
## Overview

This module contains "syntactic sugar" linear algebra routines.
_I.e._, their behaviour is relatively easily to implement in pure
PyTorch, but sometimes not very readable as it involves squeezing,
unsqueezing, or shuffling dimensions.

For example, we implement `matvec(mat, vec)`, which in PyTorch must
be written `matmul(mat, vec.unsqueeze(-1)).squeeze(-1)`.

---
"""
__all__ = [
    'kron2',
    'lmdiv',
    'rmdiv',
    'inv',
    'matvec',
    'solvevec',
    'outer',
    'trace',
    'dot',
    'mdot',
    'is_orthonormal',
    'round',
]
import torch
from torch import Tensor
from typing import Literal, Optional, Tuple


# We support pytorch >= 1.8, which has
# - torch.linalg.solve(A, B, *, out=None)
#   (note the introduction of the `left` keyword in version 1.13)
# - torch.linalg.cholesky(X, *, out=None)
# - torch.linalg.pinv(input, rcond=1e-15, hermitian=False, *, out=None)
solve = torch.linalg.solve
pinv = torch.linalg.pinv
cholesky = torch.linalg.cholesky


def kron2(a: Tensor, b: Tensor) -> Tensor:
    r"""Kronecker product of two matrices $\mathbf{A} \otimes \mathbf{B}$

    Parameters
    ----------
    a: `(..., m, n) tensor`
        Left matrix, with shape `(..., m, n)`.
    b : `(..., p, q) tensor`
        Right matrix, with shape `(..., p, q)`.

    Returns
    -------
    ab : `(..., p*m, q*n) tensor`
        Kronecker product, with shape `(..., p*m, q*n)`.

        Note: `ab.reshape([P, M, Q, N])[p, m, q, n] == a[m, n] * b[n, q]`

    References
    ----------
    - [**Kronecker product**](https://en.wikipedia.org/wiki/Kronecker_product),
    _Wikipedia_.

    """
    *_, m, n = a.shape
    *_, p, q = b.shape
    a = a[..., None, :, None, :]
    b = b[..., :, None, :, None]
    ab = a*b
    ab = ab.reshape([*ab.shape[:-4], m*p, n*q])
    return ab


def lmdiv(
    a: Tensor,
    b: Tensor,
    method: Literal['lu', 'chol', 'svd', 'pinv'] = 'lu',
    rcond: float = 1e-15,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Left matrix division $\mathbf{A}^{-1} \times \mathbf{B}$.

    !!! note
        if ``m != n``, the Moore-Penrose pseudoinverse is always used.

    Parameters
    ----------
    a : `(..., m, n) tensor`
        Left input ("the system"), with shape `(..., m, n)`.
    b : `(..., m, k) tensor`
        Right input ("the point"), with shape `(..., m, k)`.
    method : `{'lu', 'chol', 'svd', 'pinv'}`, default='lu'
        Inversion method:

        * `'lu'`   : LU decomposition. ``a`` must be invertible.
        * `'chol'` : Cholesky decomposition. ``a`` must be positive definite.
        * `'svd'`  : Singular Value decomposition.
        * `'pinv'` : Moore-Penrose pseudoinverse
                   (by means of SVD with thresholded singular values).
    rcond : `float`, default=1e-15
        Cutoff for small singular values when ``method='pinv'``.
    out : `tensor`, optional
        Output tensor.

    Returns
    -------
    x : `(..., n, k) tensor`
        Solution of the linear system, with shape `(..., n, k)`,

    References
    ----------
    1. [**LU decomposition**](https://en.wikipedia.org/wiki/LU_decomposition),
    _Wikipedia_.
    2. [**Cholesky decomposition**](https://en.wikipedia.org/wiki/Cholesky_decomposition),
    _Wikipedia_.
    3. [**Singular value decomposition**](https://en.wikipedia.org/wiki/Singular_value_decomposition),
    _Wikipedia_.
    4. [**Moore-Penrose inverse**](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse),
    _Wikipedia_.

    """
    if a.shape[-1] != a.shape[-2]:
        method = 'pinv'
    if method.lower().startswith('lu'):
        return torch.linalg.solve(a, b, out=out)
    elif method.lower().startswith('chol'):
        u = torch.linalg.cholesky(a, upper=False)
        return torch.cholesky_solve(b, u, upper=False, out=out)
    elif method.lower().startswith('svd'):
        u, s, v = torch.svd(a)
        s = s[..., None]
        return torch.matmul(v, u.transpose(-1, -2).matmul(b) / s, out=out)
    elif method.lower().startswith('pinv'):
        return torch.matmul(torch.linalg.pinv(a, rcond=rcond), b, out=out)
    else:
        raise ValueError('Unknown inversion method {}.'.format(method))


def rmdiv(
    a: Tensor,
    b: Tensor,
    method: Literal['lu', 'chol', 'svd', 'pinv'] = 'lu',
    rcond: float = 1e-15,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Right matrix division $\mathbf{A} \times \mathbf{B}^{-1}$.

    !!! note
        if ``m != n``, the Moore-Penrose pseudoinverse is always used.

    Parameters
    ----------
    a : `(..., k, m) tensor`
        Left input ("the point"), with shape `(..., k, m)`.
    b : `(..., n, m) tensor`
        Right input ("the system"), with shape `(..., n, m)`.
    method : `{'lu', 'chol', 'svd', 'pinv'}`, default='lu'
        Inversion method:

        * `'lu'`   : LU decomposition. ``a`` must be invertible.
        * `'chol'` : Cholesky decomposition. ``a`` must be positive definite.
        * `'svd'`  : Singular Value decomposition.
        * `'pinv'` : Moore-Penrose pseudoinverse
                   (by means of SVD with thresholded singular values).
    rcond : `float`, default=1e-15
        Cutoff for small singular values when ``method='pinv'``.
    out : `tensor`, optional
        Output tensor

    Returns
    -------
    x : `(..., k, m) tensor`
        Solution of the linear system, with shape `(..., k, m)`.

    References
    ----------
    1. [**LU decomposition**](https://en.wikipedia.org/wiki/LU_decomposition),
    _Wikipedia_.
    2. [**Cholesky decomposition**](https://en.wikipedia.org/wiki/Cholesky_decomposition),
    _Wikipedia_.
    3. [**Singular value decomposition**](https://en.wikipedia.org/wiki/Singular_value_decomposition),
    _Wikipedia_.
    4. [**Moore-Penrose inverse**](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse),
    _Wikipedia_.

    """
    if out is not None:
        out = out.transpose(-1, -2)
    out = lmdiv(b, a, method=method, rcond=rcond, out=out).transpose(-1, -2)
    return out


def inv(
    a: Tensor,
    method: Literal['lu', 'chol', 'svd', 'pinv'] = 'lu',
    rcond: float = 1e-15,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Matrix inversion $\mathbf{A}^{-1}$.

    !!! note
        if ``m != n``, the Moore-Penrose pseudoinverse is always used.

    Parameters
    ----------
    a : `(..., m, n) tensor`
        Input matrix, with shape `(..., m, n)`.
    method : `{'lu', 'chol', 'svd', 'pinv'}`, default='lu'
        Inversion method:

        * `'lu'`   : LU decomposition. ``a`` must be invertible.
        * `'chol'` : Cholesky decomposition. ``a`` must be positive definite.
        * `'svd'`  : Singular Value decomposition.
        * `'pinv'` : Moore-Penrose pseudoinverse
                   (by means of SVD with thresholded singular values).
    rcond : `float`, default=1e-15
        Cutoff for small singular values when ``method='pinv'``.
    out : `tensor`, optional
        Output tensor).

    Returns
    -------
    x : `(..., n, m) tensor`
        Inverse matrix, with shape `(..., n, m)`.

    References
    ----------
    1. [**LU decomposition**](https://en.wikipedia.org/wiki/LU_decomposition),
    _Wikipedia_.
    2. [**Cholesky decomposition**](https://en.wikipedia.org/wiki/Cholesky_decomposition),
    _Wikipedia_.
    3. [**Singular value decomposition**](https://en.wikipedia.org/wiki/Singular_value_decomposition),
    _Wikipedia_.
    4. [**Moore-Penrose inverse**](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse),
    _Wikipedia_.

    """
    backend = dict(dtype=a.dtype, device=a.device)
    if a.shape[-1] != a.shape[-2]:
        method = 'pinv'
    if method.lower().startswith('lu'):
        return torch.inverse(a, out=out)
    elif method.lower().startswith('chol'):
        if a.dim() == 2:
            return torch.cholesky_inverse(a, upper=False, out=out)
        else:
            chol = torch.linalg.cholesky(a, upper=False)
            eye = torch.eye(a.shape[-2], **backend)
            return torch.cholesky_solve(eye, chol, upper=False, out=out)
    elif method.lower().startswith('svd'):
        u, s, v = torch.svd(a)
        s = s[..., None]
        return v.matmul(u.transpose(-1, -2) / s)
    elif method.lower().startswith('pinv'):
        return torch.linalg.pinv(a, rcond=rcond, out=out)
    else:
        raise ValueError('Unknown inversion method {}.'.format(method))


def matvec(
        mat: Tensor,
        vec: Tensor,
        out: Optional[Tensor] = None,
) -> Tensor:
    r"""Matrix-vector product $\mathbf{A} \times \mathbf{b}$
    (supports broadcasting) .

    Parameters
    ----------
    mat : `(..., m, n) tensor`
        Input matrix, with shape `(..., m, n)`.
    vec : `(..., n) tensor`
        Input vector, with shape `(..., n)`.
    out : `(..., n) tensor`, optional
        Placeholder for the output tensor, with shape `(..., n)`.

    Returns
    -------
    mv : `(..., m) tensor`
        Matrix vector product of the inputs, with shape `(..., m)`.

    """
    vec = vec.unsqueeze(-1)
    if out is not None:
        out = out.unsqueeze(-1)
    return torch.matmul(mat, vec, out=out).squeeze(-1)


def solvevec(
    mat: Tensor,
    vec: Tensor,
    method: Literal['lu', 'chol', 'svd', 'pinv'] = 'lu',
    rcond: float = 1e-15,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Left matrix-vector division $\mathbf{A}^{-1} \times \mathbf{b}$.

    !!! note
        if ``m != n``, the Moore-Penrose pseudoinverse is always used.

    Parameters
    ----------
    mat : `(..., m, n) tensor`
        Left input ("the system"), with shape `(..., m, n)`.
    vec : `(..., m) tensor`
        Right input ("the point"), with shape `(..., m)`.
    method : `{'lu', 'chol', 'svd', 'pinv'}`, default='lu'
        Inversion method:

        * `'lu'`   : LU decomposition. ``a`` must be invertible.
        * `'chol'` : Cholesky decomposition. ``a`` must be positive definite.
        * `'svd'`  : Singular Value decomposition.
        * `'pinv'` : Moore-Penrose pseudoinverse
                   (by means of SVD with thresholded singular values).
    rcond : `float`, default=1e-15
        Cutoff for small singular values when ``method='pinv'``.
    out : `tensor`, optional
        Output tensor.

    Returns
    -------
    x : `(..., n) tensor`
        Solution of the linear system, with shape `(..., n)`,

    References
    ----------
    1. [**LU decomposition**](https://en.wikipedia.org/wiki/LU_decomposition),
    _Wikipedia_.
    2. [**Cholesky decomposition**](https://en.wikipedia.org/wiki/Cholesky_decomposition),
    _Wikipedia_.
    3. [**Singular value decomposition**](https://en.wikipedia.org/wiki/Singular_value_decomposition),
    _Wikipedia_.
    4. [**Moore-Penrose inverse**](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse),
    _Wikipedia_.

    """
    vec = vec.unsqueeze(-1)
    if out is not None:
        out = out.unsqueeze(-1)
    return lmdiv(mat, vec, method=method, rcond=rcond, out=out).squeeze(-1)


def outer(a: Tensor, b: Tensor, out: Optional[Tensor] = None) -> Tensor:
    r"""Outer product of two (batched) tensors
    $\mathbf{a} \otimes \mathbf{b} = \mathbf{a} \times \mathbf{b}^\mathrm{H}$.

    !!! warning
        When $\mathbf{a}$ and $\mathbf{b}$ are complex, this function
        returns the complex outer product
        $\mathbf{a} \otimes \mathbf{b} = \mathbf{a}\mathbf{b}^\mathrm{H}$.

    Parameters
    ----------
    a : `(..., n) tensor`
        Left vector, with shape `(..., n)`
    b : `(..., m) tensor`
        Right vector, with shape `(..., m)`
    out : `(..., n, m) tensor`, optional
        Output placeholder, with shape `(..., n, m)`

    Returns
    -------
    out : `(..., n, m) tensor`
        Outer product, with shape `(..., n, m)`

    References
    ----------
    - [**Outer product**](https://en.wikipedia.org/wiki/Outer_product),
    _Wikipedia_.

    """
    a = a.unsqueeze(-1)
    b = b.unsqueeze(-2)
    return torch.matmul(a, b.conj(), out=out)


def trace(a: Tensor, keepdim: bool = False) -> Tensor:
    r"""Compute the (batched) trace of a matrix
    $\operatorname{tr}\left(\mathbf{A}\right)$.

    Parameters
    ----------
    a : `(..., m, m) tensor`

    Returns
    -------
    t : `(..., [1, 1]) tensor`

    References
    ----------
    - [**Trace**](https://en.wikipedia.org/wiki/Trace_(linear_algebra)),
    _Wikipedia_.

    """
    t = a.diagonal(0, -1, -2).sum(-1)
    if keepdim:
        t = t[..., None, None]
    return t


def dot(
    a: Tensor,
    b: Tensor,
    keepdim: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""(Batched) dot product of two vectors $\mathbf{a}^\mathrm{H}\mathbf{b}$.

    !!! warning
        When $\mathbf{a}$ and $\mathbf{b}$ are complex, this function
        returns the complex dot product
        $(\mathbf{a}, \mathbf{b}) = \mathbf{a}^\mathrm{H}\mathbf{b}$.
        This dot product is linear in the second term and antilinear in
        the first term.

        This behaviour differs from `torch.dot` and  `torch.inner`, which
        both compute $\mathbf{a}^\mathrm{T}\mathbf{b}$ when the vectors are complex.

    Parameters
    ----------
    a : `(..., n) tensor`
        Left vector, with shape `(..., n)`
    b : `(..., m) tensor`
        Right vector, with shape `(..., m)`
    keepdim : `bool`, default=False
        Keep reduced dimension
    out : `tensor`, optional
        Output placeholder

    Returns
    -------
    ab : `(..., [1]) tensor`
        Dot product, with shape `(..., [1])`.

    References
    ----------
    - [**Dot product**](https://en.wikipedia.org/wiki/Dot_product),
    _Wikipedia_.

    """
    if out is not None:
        out = out[..., None]
        if not keepdim:
            out = out[..., None]
    a = a[..., None, :]
    b = b[..., :, None]
    out = torch.matmul(a, b, out=out)
    if keepdim:
        out = out[..., 0]
    else:
        out = out[..., 0, 0]
    return out


def mdot(
    a: Tensor,
    b: Tensor,
    keepdim: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Compute the Frobenius inner product of two matrices
    $\operatorname{tr}(\mathbf{A}^\mathrm{H}\mathbf{B})$.

    !!! warning
        When $\mathbf{A}$ and $\mathbf{B}$ are complex, this function
        returns the complex dot product
        $(\mathbf{A}, \mathbf{B}) = \operatorname{tr}(\mathbf{A}^\mathrm{H}\mathbf{B})$.
        This dot product is linear in the second term and antilinear in
        the first term.

    Parameters
    ----------
    a : `(..., n, m) tensor`
        Left matrix, with shape `(..., n, m)`.
    b : `(..., n, m) tensor`
        Right matrix, with shape `(..., n, m)`.
    keepdim : `bool`, default=False
        Keep reduced dimensions.
    out : `(..., [1, 1]) tensor`
        Output placeholder.

    Returns
    -------
    dot : `(..., [1, 1]) tensor`
        Matrix inner product, with shape `(..., [1, 1])`.

    References
    ----------
    1. [**Frobenius inner product**](https://en.wikipedia.org/wiki/Frobenius_inner_product),
        _Wikipedia_.

    """
    if out is not None and keepdim:
        out = out.squeeze(-1).squeeze(-1)
    out = dot(a.reshape(a.shape[:-2] + (-1,)),
              b.reshape(a.shape[:-2] + (-1,)), out=out)
    if keepdim:
        out = out.unsqueeze(-1).unsqueeze(-1)
    return out


def is_orthonormal(basis: Tensor, return_matrix: bool = False) \
        -> Tuple[Tensor, Optional[Tensor]]:
    """Check that a basis is an orthonormal basis.

    Parameters
    ----------
    basis : `(F, N, [M]) tensor`
        A basis of a vector or matrix space, with shape `(F, N, [M])`.
        `F` is the number of elements in the basis.
    return_matrix : `bool`, default=False
        If True, return the matrix of all pairs of inner products
        between elements if the basis

    Returns
    -------
    check : `bool`
        True if the basis is orthonormal
    matrix : `(F, F) tensor`, if `return_matrix is True`
        Matrix of all pairs of inner products, with shape `(F, F) tensor`.

    """
    basis = torch.as_tensor(basis)
    info = dict(dtype=basis.dtype, device=basis.device)
    F = basis.shape[0]
    dot = torch.dot if basis.dim() == 2 else mdot
    mat = basis.new_zeros(F, F)
    for i in range(F):
        mat[i, i] = dot(basis[i], basis[i])
        for j in range(i+1, F):
            mat[i, j] = dot(basis[i], basis[j])
            mat[j, i] = mat[i, j].conj()
    check = torch.allclose(mat, torch.eye(F, **info))
    return (check, mat) if return_matrix else check


def round(t: Tensor, decimals: int = 0) -> Tensor:
    """Round a tensor to the given number of decimals.

    Parameters
    ----------
    t : tensor
        Input tensor.
    decimals : int
        Round to this decimal.

    Returns
    -------
    t : tensor
        Rounded tensor.
    """
    return torch.round(t * 10 ** decimals) / (10 ** decimals)

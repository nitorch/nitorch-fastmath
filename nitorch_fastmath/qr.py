__all__ = [
    'eig_sym',
    'qr_hessenberg',
    'rq_hessenberg',
    'hessenberg',
    'hessenberg_sym',
    'householder',
    'householder_apply',
    'givens',
    'givens_apply',
]
import torch
from torch import Tensor
from typing import Optional, Tuple, Literal
from .typing import OneOrSeveral
from ._impl.qr import (
    eig_sym_,
    rq_hessenberg_,
    qr_hessenberg_,
    givens_apply_,
    givens,
    hessenberg_sym_upper_,
    hessenberg_sym_lower_,
    hessenberg_,
    householder_apply_,
    householder_,
)


def eig_sym(
        a: Tensor,
        compute_u: bool = False,
        upper: bool = True,
        inplace: bool = False,
        check_finite: bool = True,
        max_iter: int = 1024,
        tol: float = 1e-32,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Compute the eigendecomposition of a Hermitian square matrix.

    !!! note
        We use the explicit QR algorithm, which is probably less
        stable than the implicit QR algorithm used in Lapack.

    !!! note
        Eigenvalues are **not** sorted

        To sort eigenvalues and eigenvectors according to some criterion:
        ```python
        s, u = eig_sym(x, compute_u=True)
        _, i = crit(s).sort()
        s = s.gather(-1, i)    # permute eigenvalues
        i = i.unsqueeze(-2).expand(y.shape)
        u = u.gather(-1, i)    # permute eigenvectors
        ```

        If the criterion is the value of the eigenvalues, it simplifies:
        ```python
        s, u = eig_sym(x, compute_u=True)
        s, i = s.sort()
        i = i.unsqueeze(-2).expand(y.shape)
        u = u.gather(-1, i)    # permute eigenvectors
        ```

    Parameters
    ----------
    a : `(..., m, m) tensor`
        Input hermitian matrix or field of matrices
    compute_u : `bool`, default=False
        Compute the eigenvectors. If False, only return ``s``.
    upper : `bool`, default=True
        Whether to use the upper or lower triangular component.
    inplace : `bool`, default=False
        If `True`, overwrite ``a``.
    check_finite : `bool`, default=True
        If `True`, checks that the input matrix does not contain any
        non finite value. Disabling this may speed up the algorithm.
    max_iter : `int`, default=1024
        Maximum number of iterations.
    tol : `float`, optional
        Tolerance for early stopping.
        Default: machine precision for ``a.dtype``.

    Returns
    -------
    s : `(..., m) tensor`
        Eigenvalues.
    u : `(..., m, m) tensor`, optional
        Corresponding eigenvectors.

    """
    # Check arguments
    if check_finite and not torch.isfinite(a).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        a = a.clone()
    if a.shape[-1] != a.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(a.shape[-2], a.shape[-1]))
    return eig_sym_(a, compute_u, upper, max_iter, tol)


def rq_hessenberg(
        h: Tensor,
        u: Optional[Tensor] = None,
        inplace: bool = False,
        check_finite: bool = True,
) -> Tensor:
    """Compute the QR decomposition of a Hessenberg matrix and ``R` @ Q``.

    !!! note
        This is slower than `torch.qr` when the batch size is small,
        even though `torch.qr` does not know that ``h`` has a
        Hessenberg form. It's just hard to beat lapack. With
        larger batch size, both algorithms are on par.

    Parameters
    ----------
    h : `(..., n, n) tensor`
        Hessenberg matrix, with shape `(..., n, n)`.
        All elements should be zeros below the first lower diagonal.
    u : `(..., n) tensor`, optional
        Vectors to transform with. With shape `(..., n)`.
    inplace : `bool`, default=False
        Process inplace.
    check_finite : `bool`, default=True
        Check that all input values are finite.

    Returns
    -------
    rq : `tensor`
        Reverse product of the QR decomposition: ``rq = r @ q``.

    """
    if check_finite and not torch.isfinite(h).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        h = h.clone()
    if h.shape[-1] != h.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(h.shape[-2], h.shape[-1]))
    return rq_hessenberg_(h, u)


def qr_hessenberg(
        h: Tensor,
        inplace: bool = False,
        check_finite: bool = True,
) -> Tuple[Tensor, Tensor]:
    """QR decomposition for Hessenberg matrices.

    !!! note
        This is slower than `torch.qr` when the batch size is small,
        even though `torch.qr` does not know that ``h`` has a
        Hessenberg form. It's just hard to beat lapack. With
        larger batch size, both algorithms are on par.

    Parameters
    ----------
    h : `(..., n, n) tensor`
        Hessenberg matrix, with shape `(..., n, n)`.
        All elements should be zeros below the first lower diagonal.
    inplace : `bool`, default=False
        Process inplace.
    check_finite : `bool`, default=True
        Check that all input values are finite.

    Returns
    -------
    q : `tensor`
    r : `tensor`

    """
    if check_finite and not torch.isfinite(h).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        h = h.clone()
    if h.shape[-1] != h.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(h.shape[-2], h.shape[-1]))
    return qr_hessenberg_(h)


def hessenberg(
        a: Tensor,
        inplace: bool = False,
        check_finite: bool = True,
        compute_u: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Return an Hessenberg form of the matrix (or matrices) ``a``.

    Parameters
    ----------
    a : `(..., n, n) tensor`
        Input matrix, with shape `(..., n, n)`. Can be complex.
    inplace : `bool`, default=False
        Overwrite ``a``.
    check_finite : `bool`, default=True
        Check that all values in ``a`` ar finite.
    compute_u : `bool`, default=False
        Compute and return the transformation matrix ``u``.

    Returns
    -------
    h : `(..., n, n) tensor`
        Hessenberg form of ``a``, with shape `(..., n, n)`.
    u : `list[tensor]`, optional
        Set of Householder reflectors.

    References
    ----------
    - [**Hessenberg matrix**](https://en.wikipedia.org/wiki/Hessenberg_matrix),
    _Wikipedia_

    """
    if check_finite and not torch.isfinite(a).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        a = a.clone()
    if a.shape[-1] != a.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(a.shape[-2], a.shape[-1]))
    return hessenberg_(a, compute_u)


def hessenberg_sym(
        a: Tensor,
        upper: bool = True,
        fill: bool = True,
        inplace: bool = False,
        check_finite: bool = True,
        compute_u: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Return a tridiagonal form of the hermitian matrix (or matrices) ``a``.

    !!! note
        The Hessenberg form of a Hermitian matrix is tridiagonal.

    Parameters
    ----------
    a : `(..., n, n) tensor`
        Input hermitian matrix, with shape (..., n, n)`. Can be complex.
    upper : `bool`, default=True
        Whether to use the upper or lower triangular part of the matrix.
    fill : `bool`, default=False
        Fill the other half of the output matrix by conjugate symmetry.
    inplace : `bool`, default=False
        Overwrite ``a``.
    check_finite : `bool`, default=True
        Check that all values in ``a`` ar finite.
    compute_u : `bool`, default=False
        Compute and return the transformation matrix ``u``.

    Returns
    -------
    h : `(..., n, n) tensor`
        Hessenberg (tridiagonal) form of ``a``, with shape `(..., n, n)`.
    u : `list[tensor]`, optional
        Set of Householder reflectors.

    References
    ----------
    - [**Hessenberg matrix**](https://en.wikipedia.org/wiki/Hessenberg_matrix),
    _Wikipedia_

    """
    if check_finite and not torch.isfinite(a).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        a = a.clone()
    if a.shape[-1] != a.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(a.shape[-2], a.shape[-1]))
    return (hessenberg_sym_upper_(a, compute_u, fill) if upper else
            hessenberg_sym_lower_(a, compute_u, fill))


def householder(
        x: Tensor,
        basis: int = 0,
        inplace: bool = False,
        check_finite: bool = True,
        return_alpha: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Compute the Householder reflector of a (batch of) vector(s).

    !!! note
        Householder reflectors are matrices of the form
        $\mathbf{P} = \mathbf{I} - 2\mathbf{u}\mathbf{u}^\ast$,
        where $\mathbf{u}$ is a Householder vector. Reflectors are typically used
        to project a (complex) vector onto a Euclidean basis:
        $\mathbf{P} \times \mathbf{x} = r \lVert\mathbf{x}\rVert \mathbf{e}_i$,
        with $r = -\exp(i*\operatorname{angle}(x_i))$. This function
        returns a Householder vector tailored to a specific (complex) vector.


    Parameters
    ----------
    x : `(..., n) tensor`
        Input vector, with shape `(..., n)` where `...` is any number
        of leading batch dimensions.
    basis : `int`, default=1
        Index of the Euclidean basis.
    inplace : `bool`, default=False
        If True, overwrite `x`.
    check_finite : `bool`, default=True
        If True, checks that the input matrix does not contain any
        non finite value. Disabling this may speed up the algorithm.
    return_alpha : `bool`, default=False
        Return alpha, the 'projection' of x on the Euclidean basis:
        $-\lVert \mathbf{x} \rVert \times \exp(i*\operatorname{angle}(x_i))$,
        where $i$ is the index of the Euclidean basis.

    Returns
    -------
    u : `(..., n) tensor`
        Householder vector, with shape `(..., n)`.
    a : `(...) tensor`, optional
        Projection on the Euclidean basis, with shape `(...)`.

    """
    if check_finite and not torch.isfinite(x).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        x = x.clone()
    x, alpha = householder_(x, basis)
    return (x, alpha) if return_alpha else x


def householder_apply(
        a: Tensor,
        u: Tensor,
        k: Optional[OneOrSeveral[int]] = None,
        side: Literal['left', 'right', 'both'] = 'both',
        inverse: bool = False,
        inplace: bool = False,
        check_finite: bool = True,
) -> Tensor:
    r"""Apply a series of Householder reflectors to a matrix.

    Parameters
    ----------
    a : `(..., n, n) tensor`
        $N \times N$ matrix, with shape `(..., n, n)`
        where `...` is any number of leading batch dimensions.
    u : `tensor or list[tensor]`
        A list of Householder reflectors $\left\{\mathbf{u}_k\right\}_{k=1}^K$.
        Each reflector forms the Householder matrix
        $\mathbf{P}_k = \mathbf{I} - 2 \mathbf{u}_k \mathbf{u}_k^H$.
    k : `int or list[int]`, optional
        The index corresponding to each reflector.
    side : `{'left', 'right', 'both'}`, default='both'
        Side to apply to.
    inverse : `bool`, default=False
        Apply the inverse transform
    inplace : `bool`, default=False
        Apply transformation inplace.
    check_finite : `bool`, default=True
        Check that the input does not have any nonfinite values

    Returns
    -------
    h : `(..., n, n) tensor`
        The transformed matrix: $\mathbf{H} = \mathbf{U} \times \mathbf{A} \times \mathbf{U}^H$
        with $\mathbf{U} = \mathbf{P}_{k-2} \times ... \times \mathbf{P}_1$.

    """
    if check_finite and not torch.isfinite(a).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        a = a.clone()
    return householder_apply_(a, u, k, side, inverse)


def givens_apply(
        a: Tensor,
        c: Tensor,
        s: Tensor,
        i: int = 0,
        j: Optional[int] = None,
        side: Literal['left', 'right', 'both'] = 'both',
        inplace: bool = False,
        check_finite: bool = True,
) -> Tensor:
    """ Apply a Givens rotation to a matrix

    Parameters
    ----------
    a : `(..., n, n) tensor`
        Input matrix
    c : `tensor`
        Cosine of the Givens rotation
    s : `tensor`
        Sine of the Givens rotation
    i : int
        Index in `a` of the first component of the 2D vector to rotate
    j : `int`, default=`i+1`
        Index in `a` of the second component of the 2D vector to rotate
    side : `{'left', 'right', 'both'}`, default='both'
        Whether to apply the rotation on the left or on the right.
    inplace : `bool`
        Apply the rotation in-place.
    check_finite : `bool`
        Check for non-finite values

    Returns
    -------
    a : `(..., n, n) tensor`
        Rotated matrix

    References
    ----------
    - [**Givens rotation**](https://en.wikipedia.org/wiki/Givens_rotation),
      _Wikipedia_

    """
    if check_finite and not torch.isfinite(a).all():
        raise ValueError('Input has non finite values.')
    if not inplace:
        a = a.clone()
    if a.shape[-1] != a.shape[-2]:
        raise ValueError('Expected square matrix. Got ({}, {})'
                         .format(a.shape[-2], a.shape[-1]))
    return givens_apply_(a, c, s, i, j, side)
"""Linear algebra."""
import torch
from warnings import warn
import math
from nitorch_core import py
from nitorch_core.tensors import as_tensor, max_backend, to_max_backend


# Expose from private implementation
from ._expm import expm, _expm
from ._logm import logm
from ._linalg_qr import eig_sym, eig_sym_
from .sym_script import *


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'solve'):
    _solve_lu = torch.linalg.solve
else:
    _solve_lu = lambda A, b: torch.solve(b, A)[0]


def meanm(mats, max_iter=1024, tol=1e-20):
    """Compute the exponential barycentre of a set of matrices.

    Parameters
    ----------
    mats : (N, M, M) tensor_like[float]
        Set of square invertible matrices
    max_iter : int, default=1024
        Maximum number of iterations
    tol : float, default=1E-20
        Tolerance for early stopping.
        The tolerance criterion is the sum-of-squares of the residuals
        in log-space, _i.e._, :math:`||\sum_n \log_M(A_n) / N||^2`

    Returns
    -------
    mean_mat : (M, M) tensor
        Mean matrix.

    References
    ----------
    .. [1]  Xavier Pennec, Vincent Arsigny.
        "Exponential Barycenters of the Canonical Cartan Connection and
        Invariant Means on Lie Groups."
        Matrix Information Geometry, Springer, pp.123-168, 2012.
        (10.1007/978-3-642-30232-9_7). (hal-00699361)
        https://hal.inria.fr/hal-00699361

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2012-2019 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    # NOTE: all computations are performed in double, else logm is not
    # precise enough

    mats = as_tensor(mats)
    dim = mats.shape[-1] - 1
    dtype = mats.dtype
    device = mats.device
    mats = mats.double()

    mean_mat = torch.eye(dim+1, dtype=torch.double, device=device)
    for n_iter in range(max_iter):
        # Project all matrices to the tangent space about the current mean_mat
        log_mats = lmdiv(mean_mat, mats)
        log_mats = logm(log_mats)
        if log_mats.is_complex():
            warn('`meanm` failed to converge (`logm` -> complex)',
                 RuntimeWarning)
            break
        # Compute the new mean in the tangent space
        mean_log_mat = torch.mean(log_mats, dim=0)
        # Compute sum-of-squares in tangent space (should be zero at optimum)
        sos = mean_log_mat.square().sum()
        # Exponentiate to original space
        mean_mat = torch.matmul(mean_mat, expm(mean_log_mat))
        if sos <= tol:
            break

    return mean_mat.to(dtype)


def kron2(x, y):
    """Kronecker product of two matrices

    Parameters
    ----------
    x : (..., m, n) tensor
        Left matrix
    y : (..., p, q) tensor
        Right matrix

    Returns
    -------
    xy : (..., p*m, q*n) tensor
        Kronecker product
        `xy.reshape([P, M, Q, N])[p, m, q, n] == x[m, n] * y[n, q]`

    """
    x, y = to_max_backend(x, y)
    *_, m, n = x.shape
    *_, p, q = y.shape
    x = x[..., None, :, None, :]
    y = y[..., :, None, :, None]
    xy = x*y
    xy = xy.reshape([*xy.shape[:-4], m*p, n*q])
    return xy


def lmdiv(a, b, method='lu', rcond=1e-15, out=None):
    r"""Left matrix division ``inv(a) @ b``.

    Parameters
    ----------
    a : (..., m, n) tensor_like
        Left input ("the system")
    b : (..., m, k) tensor_like
        Right input ("the point")
    method : {'lu', 'chol', 'svd', 'pinv'}, default='lu'
        Inversion method:
        * 'lu'   : LU decomposition. ``a`` must be invertible.
        * 'chol' : Cholesky decomposition. ``a`` must be positive definite.
        * 'svd'  : Singular Value decomposition.
        * 'pinv' : Moore-Penrose pseudoinverse (by means of svd).
    rcond : float, default=1e-15
        Cutoff for small singular values when ``method == 'pinv'``.
    out : tensor, optional
        Output tensor (only used by methods 'lu' and 'chol').

    .. note:: if ``m != n``, the Moore-Penrose pseudoinverse is always used.

    Returns
    -------
    x : (..., n, k) tensor
        Solution of the linear system.

    """
    backend = max_backend(a, b)
    a = as_tensor(a, **backend)
    b = as_tensor(b, **backend)
    if a.shape[-1] != a.shape[-2]:
        method = 'pinv'
    if method.lower().startswith('lu'):
        # TODO: out keyword
        return _solve_lu(a, b)
    elif method.lower().startswith('chol'):
        u = torch.cholesky(a, upper=False)
        return torch.cholesky_solve(b, u, upper=False, out=out)
    elif method.lower().startswith('svd'):
        u, s, v = torch.svd(a)
        s = s[..., None]
        return v.matmul(u.transpose(-1, -2).matmul(b) / s)
    elif method.lower().startswith('pinv'):
        return torch.pinverse(a, rcond=rcond).matmul(b)
    else:
        raise ValueError('Unknown inversion method {}.'.format(method))


def rmdiv(a, b, method='lu', rcond=1e-15, out=None):
    r"""Right matrix division ``a @ inv(b)``.

    Parameters
    ----------
    a : (..., k, m) tensor_like
        Left input ("the point")
    b : (..., n, m) tensor_like
        Right input ("the system")
    method : {'lu', 'chol', 'svd', 'pinv'}, default='lu'
        Inversion method:
        * 'lu'   : LU decomposition. ``a`` must be invertible.
        * 'chol' : Cholesky decomposition. ``a`` must be positive definite.
        * 'svd'  : Singular Value decomposition. ``a`` must be invertible.
        * 'pinv' : Moore-Penrose pseudoinverse
                   (by means of svd with thresholded singular values).
    rcond : float, default=1e-15
        Cutoff for small singular values when ``method == 'pinv'``.
    out : tensor, optional
        Output tensor (only used by methods 'lu' and 'chol').

    .. note:: if ``m != n``, the Moore-Penrose pseudoinverse is always used.

    Returns
    -------
    x : (..., k, m) tensor
        Solution of the linear system.

    """
    backend = max_backend(a, b)
    a = as_tensor(a, **backend).transpose(-1, -2)
    b = as_tensor(b, **backend).transpose(-1, -2)
    x = lmdiv(b, a, method=method, rcond=rcond).transpose(-1, -2)
    return x


def inv(a, method='lu', rcond=1e-15, out=None):
    r"""Matrix inversion.

    Parameters
    ----------
    a : (..., m, n) tensor_like
        Input matrix.
    method : {'lu', 'chol', 'svd', 'pinv'}, default='lu'
        Inversion method:
        * 'lu'   : LU decomposition. ``a`` must be invertible.
        * 'chol' : Cholesky decomposition. ``a`` must be positive definite.
        * 'svd'  : Singular Value decomposition.
        * 'pinv' : Moore-Penrose pseudoinverse (by means of svd).
    rcond : float, default=1e-15
        Cutoff for small singular values when ``method == 'pinv'``.
    out : tensor, optional
        Output tensor (only used by methods 'lu' and 'chol').

    .. note:: if ``m != n``, the Moore-Penrose pseudoinverse is always used.

    Returns
    -------
    x : (..., n, m) tensor
        Inverse matrix.

    """
    a = as_tensor(a)
    backend = dict(dtype=a.dtype, device=a.device)
    if a.shape[-1] != a.shape[-2]:
        method = 'pinv'
    if method.lower().startswith('lu'):
        return torch.inverse(a, out=out)
    elif method.lower().startswith('chol'):
        if a.dim() == 2:
            return torch.cholesky_inverse(a, upper=False, out=out)
        else:
            chol = torch.cholesky(a, upper=False)
            eye = torch.eye(a.shape[-2], **backend)
            return torch.cholesky_solve(eye, chol, upper=False, out=out)
    elif method.lower().startswith('svd'):
        u, s, v = torch.svd(a)
        s = s[..., None]
        return v.matmul(u.transpose(-1, -2) / s)
    elif method.lower().startswith('pinv'):
        return torch.pinverse(a, rcond=rcond)
    else:
        raise ValueError('Unknown inversion method {}.'.format(method))


def matvec(mat, vec, out=None):
    """Matrix-vector product (supports broadcasting)

    Parameters
    ----------
    mat : (..., M, N) tensor
        Input matrix.
    vec : (..., N) tensor
        Input vector.
    out : (..., M) tensor, optional
        Placeholder for the output tensor.

    Returns
    -------
    mv : (..., M) tensor
        Matrix vector product of the inputs

    """
    mat = torch.as_tensor(mat)
    vec = torch.as_tensor(vec)[..., None]
    if out is not None:
        out = out[..., None]

    mv = torch.matmul(mat, vec, out=out)
    mv = mv[..., 0]
    if out is not None:
        out = out[..., 0]

    return mv


def outer(a, b, out=None):
    """Outer product of two (batched) tensors

    Parameters
    ----------
    a : (..., N) tensor
    b : (..., M) tensor
    out : (..., N, M) tensor, optional

    Returns
    -------
    out : (..., N, M) tensor

    """
    a = a.unsqueeze(-1)
    b = b.unsqueeze(-2)
    return torch.matmul(a, b)


def dot(a, b, keepdim=False, out=None):
    """(Batched) dot product

    Parameters
    ----------
    a : (..., N) tensor
    b : (..., N) tensor
    keepdim : bool, default=False
    out : tensor, optional

    Returns
    -------
    ab : (..., [1]) tensor

    """
    a = a[..., None, :]
    b = b[..., :, None]
    ab = torch.matmul(a, b, out=out)
    if keepdim:
        ab = ab[..., 0]
    else:
        ab = ab[..., 0, 0]
    return ab


def cholesky(a):
    """Compute the Choleksy decomposition of a positive-definite matrix

    Parameters
    ----------
    a : (..., M, M) tensor
        Positive-definite matrix

    Returns
    -------
    l : (..., M, M) tensor
        Lower-triangular cholesky factor

    """
    if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'cholesky'):
        return torch.linalg.cholesky(a)
    return torch.cholesky(a)


def trace(a, keepdim=False):
    """Compute the trace of a matrix (or batch)

    Parameters
    ----------
    a : (..., M, M) tensor

    Returns
    -------
    t : (...) tensor

    """
    t = a.diagonal(0, -1, -2).sum(-1)
    if keepdim:
        t = t[..., None, None]
    return t


def mdot(a, b):
    """Compute the Frobenius inner product of two matrices

    Parameters
    ----------
    a : (..., N, M) tensor
        Left matrix
    b : (..., N, M) tensor
        Rightmatrix

    Returns
    -------
    dot : (...) tensor
        Matrix inner product

    References
    ----------
    ..[1] https://en.wikipedia.org/wiki/Frobenius_inner_product

    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    mm = torch.matmul(a.conj().transpose(-1, -2), b)
    if a.dim() == b.dim() == 2:
        return mm.trace()
    else:
        return mm.diagonal(0, -1, -2).sum(dim=-1)


def is_orthonormal(basis, return_matrix=False):
    """Check that a basis is an orthonormal basis.

    Parameters
    ----------
    basis : (F, N, [M])
        A basis of a vector or matrix space.
        `F` is the number of elements in the basis.
    return_matrix : bool, default=False
        If True, return the matrix of all pairs of inner products
        between elements if the basis

    Returns
    -------
    check : bool
        True if the basis is orthonormal
    matrix : (F, F) tensor, if `return_matrix is True`
        Matrix of all pairs of inner products

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


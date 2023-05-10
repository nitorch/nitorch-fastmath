"""
Linear Algebra utilities.

I found that some torch functions (e.g., `inverse()` or `det()`) were
not so efficient when applied to large batches of small matrices,
especially on the GPU (this is not so true on the CPU). I reimplemented
them using torchscript for 2x2 and 3x3 matrices, and they are much
faster:
    - batchdet
    - batchinv
    - batchmatvec
I used to have a `batchmatmul` too, but its speed was not always better
than `torch.matmul()` (it depended a lot on the striding layout),
so I removed it.
"""

import torch
from ..sugar import matvec


@torch.jit.script
def det2(a):
    dt = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    return dt


@torch.jit.script
def det3(a):
    dt = a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) + \
         a[0, 1] * (a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2]) + \
         a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0])
    return dt


def batchdet(a):
    """Efficient batched determinant for large batches of small matrices

    !!! note
        A batched implementation is used for 1x1, 2x2 and 3x3 matrices.
        Other sizes fall back to `torch.det`.

    Parameters
    ----------
    a : (..., n, n) tensor
        Input matrix.

    Returns
    -------
    d : (...) tensor
        Determinant.

    """
    if not a.is_cuda or a.shape[-1] > 3:
        return a.det()
    a = a.movedim(-1, 0).movedim(-1, 0)
    if len(a) == 3:
        a = det3(a)
    elif len(a) == 2:
        a = det2(a)
    else:
        assert len(a) == 1
        a = a.clone()[0, 0]
    return a


@torch.jit.script
def inv2(A):
    F = torch.empty_like(A)
    F[0, 0] = A[1, 1]
    F[1, 1] = A[0, 0]
    F[0, 1] = -A[1, 0]
    F[1, 0] = -A[0, 1]
    dt = det2(A)
    Aabs = A.reshape((-1,) + A.shape[2:]).abs()
    rnge = Aabs.max(dim=0).values - Aabs.min(dim=0).values
    dt += rnge * 1E-12
    F /= dt[None, None]
    return F


@torch.jit.script
def inv3(A):
    F = torch.empty_like(A)
    F[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    F[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    F[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    F[0, 1] = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    F[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    F[1, 0] = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    F[1, 2] = A[1, 0] * A[0, 2] - A[1, 2] * A[0, 0]
    F[2, 0] = A[2, 1] * A[1, 0] - A[2, 0] * A[1, 1]
    F[2, 1] = A[2, 0] * A[0, 1] - A[2, 1] * A[0, 0]
    dt = det3(A)
    Aabs = A.reshape((-1,) + A.shape[2:]).abs()
    rnge = Aabs.max(dim=0).values - Aabs.min(dim=0).values
    dt += rnge * 1E-12
    F /= dt[None, None]
    return F


def batchinv(a):
    """Efficient batched inversion for large batches of small matrices

    !!! note
        A batched implementation is used for 1x1, 2x2 and 3x3 matrices.
        Other sizes fall back to `torch.linagl.inv`.

    Parameters
    ----------
    a : (..., n, n) tensor
        Input matrix.

    Returns
    -------
    a : (..., n, n) tensor
        Inverse matrix.

    """
    if not a.is_cuda or a.shape[-1] > 3:
        return a.inverse()
    a = a.movedim(-1, 0).movedim(-1, 0)
    if len(a) == 3:
        a = inv3(a)
    elif len(a) == 2:
        a = inv2(a)
    else:
        assert len(a) == 1
        a = a.reciprocal()
    a = a.movedim(0, -1).movedim(0, -1)
    return a


@torch.jit.script
def matvec3(A, v, Av):
    Av[0] = A[0, 0] * v[0] + A[0, 1] * v[1] + A[0, 2] * v[2]
    Av[1] = A[1, 0] * v[0] + A[1, 1] * v[1] + A[1, 2] * v[2]
    Av[2] = A[2, 0] * v[0] + A[2, 1] * v[1] + A[2, 2] * v[2]
    return Av


@torch.jit.script
def matvec2(A, v, Av):
    Av[0] = A[0, 0] * v[0] + A[0, 1] * v[1]
    Av[1] = A[1, 0] * v[0] + A[1, 1] * v[1]
    return Av


@torch.jit.script
def matvec1(A, v, Av):
    Av[0] = A[0, 0] * v[0]
    return Av


def batchmatvec(mat, vec):
    """Efficient batched matrix-vector product for large batches of small matrices

    !!! note
        A batched implementation is used for 1x1, 2x2 and 3x3 matrices.
        Other sizes fall back to `matvec`.

    Parameters
    ----------
    mat : (..., m, n) tensor
        Input matrix.
    vec : (..., n) tensor
        Input vector.

    Returns
    -------
    matvec : (..., m) tensor
        Matrix-vector product.

    """
    m, n = mat.shape[-2:]
    if not mat.is_cuda or (m != n) or (n > 3):
        return matvec(mat, vec)
    vec = vec.movedim(-1, 0)
    mat = mat.movedim(-1, 0).movedim(-1, 0)
    batch = torch.broadcast_shapes(mat.shape[2:], vec.shape[1:])
    out = vec.new_empty(vec.shape[:1] + batch)
    if n == 1:
        mv = matvec1
    elif n == 2:
        mv = matvec2
    else:
        assert n == 3
        mv = matvec3
    out = mv(mat, vec, out)
    out = out.movedim(0, -1)
    return out
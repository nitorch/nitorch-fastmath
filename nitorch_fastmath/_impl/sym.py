__all__ = [
    'sym_to_full', 'sym_diag', 'sym_outer', 'sym_det',
    'sym_invert', 'sym_solve', 'sym_matvec']
import torch
from torch import Tensor
from typing import List, Optional
from math import sqrt


if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'solve'):
    _solve_lu = torch.linalg.solve
else:
    _solve_lu = lambda A, b: torch.solve(b, A)[0]


def sym_to_full(mat: Tensor) -> Tensor:
    r"""Transform a symmetric matrix into a full matrix

    Parameters
    ----------
    mat : `(..., M*(M+1)//2) tensor`
        A $M \times M$  symmetric matrix that is stored in a compact way,
        with shape `(..., M*(M+1)//2)`, where `...` is any number of
        leading batch dimensions.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., `[a00, a11, aa22, a01, a02, a12]`.

    Returns
    -------
    full : `(..., M, M) tensor`
        Full matrix

    """
    mat = torch.as_tensor(mat)
    mat = mat.movedim(-1, 0)
    nb_prm = int((sqrt(1 + 8 * len(mat)) - 1)//2)
    full = mat.new_empty([nb_prm, nb_prm, *mat.shape[1:]])
    i = 0
    for i in range(nb_prm):
        full[i, i].copy_(mat[i])
    count = i + 1
    for i in range(nb_prm):
        for j in range(i+1, nb_prm):
            full[i, j].copy_(mat[count])
            full[j, i].copy_(mat[count])
            count += 1

    # full = [[None] * nb_prm for _ in range(nb_prm)]
    # i = 0
    # for i in range(nb_prm):
    #     full[i][i] = mat[i]
    # count = i + 1
    # for i in range(nb_prm):
    #     for j in range(i+1, nb_prm):
    #         full[i][j] = full[j][i] = mat[count]
    #         count += 1
    # full = as_tensor(full)

    return full.movedim(0, -1).movedim(0, -1)


def sym_diag(mat: Tensor) -> Tensor:
    r"""Diagonal of a symmetric matrix

    Parameters
    ----------
    mat : `(..., M * (M+1) // 2) tensor`
        A $M \times M$  symmetric matrix that is stored in a compact way,
        with shape `(..., M*(M+1)//2)`, where `...` is any number of
        leading batch dimensions.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., `[a00, a11, aa22, a01, a02, a12]`.

    Returns
    -------
    diag : `(..., M) tensor`
        View into the main diagonal of the matrix, with shape `(..., M)`.

    """
    mat = torch.as_tensor(mat)
    nb_prm = int((sqrt(1 + 8 * mat.shape[-1]) - 1)//2)
    return mat[..., :nb_prm]


@torch.jit.script
def _sym_matvec2(mat, vec):
    mm = mat[:2] * vec
    mm[0].addcmul_(mat[2], vec[1])
    mm[1].addcmul_(mat[2], vec[0])
    return mm


@torch.jit.script
def _sym_matvec3(mat, vec):
    mm = mat[:3] * vec
    mm[0].addcmul_(mat[3], vec[1]).addcmul_(mat[4], vec[2])
    mm[1].addcmul_(mat[3], vec[0]).addcmul_(mat[5], vec[2])
    mm[2].addcmul_(mat[4], vec[0]).addcmul_(mat[5], vec[1])
    return mm


@torch.jit.script
def _sym_matvec4(mat, vec):
    mm = mat[:4] * vec
    mm[0].addcmul_(mat[4], vec[1]) \
        .addcmul_(mat[5], vec[2]) \
        .addcmul_(mat[6], vec[3])
    mm[1].addcmul_(mat[4], vec[0]) \
        .addcmul_(mat[7], vec[2]) \
        .addcmul_(mat[8], vec[3])
    mm[2].addcmul_(mat[5], vec[0]) \
        .addcmul_(mat[7], vec[1]) \
        .addcmul_(mat[9], vec[3])
    mm[3].addcmul_(mat[6], vec[0]) \
        .addcmul_(mat[8], vec[1]) \
        .addcmul_(mat[9], vec[2])
    return mm


@torch.jit.script
def _sym_matvecn(mat, vec, nb_prm: int):
    mm = mat[:nb_prm] * vec
    c = nb_prm
    for i in range(nb_prm):
        for j in range(i+1, nb_prm):
            mm[i].addcmul_(mat[c], vec[j])
            mm[j].addcmul_(mat[c], vec[i])
            c += 1
    return mm


def sym_matvec(mat: Tensor, vec: Tensor) -> Tensor:
    r"""Matrix-vector product with a symmetric matrix

    Parameters
    ----------
    mat : `(..., M*(M+1)//2) tensor`
        A $M \times M$  symmetric matrix that is stored in a compact way,
        with shape `(..., M*(M+1)//2)`, where `...` is any number of
        leading batch dimensions.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., `[a00, a11, aa22, a01, a02, a12]`.
    vec : `(..., M) tensor`
        A vector, with shape `(..., M)`.

    Returns
    -------
    matvec : `(..., M) tensor`
        The matrix-vector product, with shape `(..., M)`.

    """
    nb_prm = vec.shape[-1]
    if nb_prm == 1:
        return mat * vec

    # make the vector dimension first so that the code is less ugly
    mat = torch.movedim(mat, -1, 0)
    vec = torch.movedim(vec, -1, 0)

    if nb_prm == 2:
        mm = _sym_matvec2(mat, vec)
    elif nb_prm == 3:
        mm = _sym_matvec3(mat, vec)
    elif nb_prm == 4:
        mm = _sym_matvec4(mat, vec)
    else:
        mm = _sym_matvecn(mat, vec, nb_prm)

    return torch.movedim(mm, 0, -1)


@torch.jit.script
def _square(x):
    return x * x


@torch.jit.script
def _square_(x):
    x *= x
    return x


@torch.jit.script
def _sym_det2(diag, uppr):
    det = _square(uppr[0]).neg_()
    det.addcmul_(diag[0], diag[1])
    return det


@torch.jit.script
def _sym_solve2(diag, uppr, vec, shape: List[int]):
    det = _sym_det2(diag, uppr)
    res = vec.new_empty(shape)
    res[0] = diag[1] * vec[0] - uppr[0] * vec[1]
    res[1] = diag[0] * vec[1] - uppr[0] * vec[0]
    res /= det
    return res


@torch.jit.script
def _sym_det3(diag, uppr):
    det = diag.prod(0) + 2 * uppr.prod(0) \
        - (diag[0] * _square(uppr[2]) +
           diag[2] * _square(uppr[0]) +
           diag[1] * _square(uppr[1]))
    return det


@torch.jit.script
def _sym_solve3(diag, uppr, vec, shape: List[int]):
    det = _sym_det3(diag, uppr)
    res = vec.new_empty(shape)
    res[0] = (diag[1] * diag[2] - _square(uppr[2])) * vec[0] \
           + (uppr[1] * uppr[2] - diag[2] * uppr[0]) * vec[1] \
           + (uppr[0] * uppr[2] - diag[1] * uppr[1]) * vec[2]
    res[1] = (uppr[1] * uppr[2] - diag[2] * uppr[0]) * vec[0] \
           + (diag[0] * diag[2] - _square(uppr[1])) * vec[1] \
           + (uppr[0] * uppr[1] - diag[0] * uppr[2]) * vec[2]
    res[2] = (uppr[0] * uppr[2] - diag[1] * uppr[1]) * vec[0] \
           + (uppr[0] * uppr[1] - diag[0] * uppr[2]) * vec[1] \
           + (diag[0] * diag[1] - _square(uppr[0])) * vec[2]
    res /= det
    return res


@torch.jit.script
def _sym_det4(diag, uppr):
    det = diag.prod(0) \
         + (_square(uppr[0] * uppr[5]) +
            _square(uppr[1] * uppr[4]) +
            _square(uppr[2] * uppr[3])) + \
         - 2 * (uppr[0] * uppr[1] * uppr[4] * uppr[5] +
                uppr[0] * uppr[2] * uppr[3] * uppr[5] +
                uppr[1] * uppr[2] * uppr[3] * uppr[4]) \
         + 2 * (diag[0] * uppr[3] * uppr[4] * uppr[5] +
                diag[1] * uppr[1] * uppr[2] * uppr[5] +
                diag[2] * uppr[0] * uppr[2] * uppr[4] +
                diag[3] * uppr[0] * uppr[1] * uppr[3]) \
         - (diag[0] * diag[1] * _square(uppr[5]) +
            diag[0] * diag[2] * _square(uppr[4]) +
            diag[0] * diag[3] * _square(uppr[3]) +
            diag[1] * diag[2] * _square(uppr[2]) +
            diag[1] * diag[3] * _square(uppr[1]) +
            diag[2] * diag[3] * _square(uppr[0]))
    return det


@torch.jit.script
def _sym_solve4(diag, uppr, vec, shape: List[int]):
    det = _sym_det4(diag, uppr)
    inv01 = (- diag[2] * diag[3] * uppr[0]
             + diag[2] * uppr[2] * uppr[4]
             + diag[3] * uppr[1] * uppr[3]
             + uppr[0] * _square(uppr[5])
             - uppr[1] * uppr[4] * uppr[5]
             - uppr[2] * uppr[3] * uppr[5])
    inv02 = (- diag[1] * diag[3] * uppr[1]
             + diag[1] * uppr[2] * uppr[5]
             + diag[3] * uppr[0] * uppr[3]
             + uppr[1] * _square(uppr[4])
             - uppr[0] * uppr[4] * uppr[5]
             - uppr[2] * uppr[3] * uppr[4])
    inv03 = (- diag[1] * diag[2] * uppr[2]
             + diag[1] * uppr[1] * uppr[5]
             + diag[2] * uppr[0] * uppr[4]
             + uppr[2] * _square(uppr[3])
             - uppr[0] * uppr[3] * uppr[5]
             - uppr[1] * uppr[3] * uppr[4])
    inv12 = (- diag[0] * diag[3] * uppr[3]
             + diag[0] * uppr[4] * uppr[5]
             + diag[3] * uppr[0] * uppr[1]
             + uppr[3] * _square(uppr[2])
             - uppr[0] * uppr[2] * uppr[5]
             - uppr[1] * uppr[2] * uppr[4])
    inv13 = (- diag[0] * diag[2] * uppr[4]
             + diag[0] * uppr[3] * uppr[5]
             + diag[2] * uppr[0] * uppr[2]
             + uppr[4] * _square(uppr[1])
             - uppr[0] * uppr[1] * uppr[5]
             - uppr[1] * uppr[2] * uppr[3])
    inv23 = (- diag[0] * diag[1] * uppr[5]
             + diag[0] * uppr[4] * uppr[3]
             + diag[1] * uppr[1] * uppr[2]
             + uppr[5] * _square(uppr[0])
             - uppr[0] * uppr[1] * uppr[4]
             - uppr[0] * uppr[2] * uppr[3])
    res = vec.new_empty(shape)
    res[0] = (diag[1] * diag[2] * diag[3]
              - diag[1] * _square(uppr[5])
              - diag[2] * _square(uppr[4])
              - diag[3] * _square(uppr[3])
              + 2 * uppr[3] * uppr[4] * uppr[5]) * vec[0]
    res[0] += inv01 * vec[1]
    res[0] += inv02 * vec[2]
    res[0] += inv03 * vec[3]
    res[1] = (diag[0] * diag[2] * diag[3]
              - diag[0] * _square(uppr[5])
              - diag[2] * _square(uppr[2])
              - diag[3] * _square(uppr[1])
              + 2 * uppr[1] * uppr[2] * uppr[5]) * vec[1]
    res[1] += inv01 * vec[0]
    res[1] += inv12 * vec[2]
    res[1] += inv13 * vec[3]
    res[2] = (diag[0] * diag[1] * diag[3]
              - diag[0] * _square(uppr[4])
              - diag[1] * _square(uppr[2])
              - diag[3] * _square(uppr[0])
              + 2 * uppr[0] * uppr[2] * uppr[4]) * vec[2]
    res[2] += inv02 * vec[0]
    res[2] += inv12 * vec[1]
    res[2] += inv23 * vec[3]
    res[3] = (diag[0] * diag[1] * diag[2]
              - diag[0] * _square(uppr[3])
              - diag[1] * _square(uppr[1])
              - diag[2] * _square(uppr[0])
              + 2 * uppr[0] * uppr[1] * uppr[3]) * vec[3]
    res[3] += inv03 * vec[0]
    res[3] += inv13 * vec[1]
    res[3] += inv23 * vec[2]
    res /= det
    return res


def sym_solve(mat: Tensor, vec: Tensor, eps: Optional[float] = None) -> Tensor:
    r"""Left matrix division for compact symmetric matrices.

    `>>> mat \ vec`

    Warning
    -------
    - Currently, autograd does not work through this function.
    - The order of arguments is the inverse of torch.solve

    Notes
    -----
    - Orders up to 4 are implemented in closed-form.
    - Orders > 4 use torch's batched implementation but require
      building the full matrices.
    - Backpropagation works at least for torch >= 1.6
      It should be checked on earlier versions.

    Parameters
    ----------
    mat : `(..., M*(M+1)//2) tensor`
        A $M \times M$  symmetric matrix that is stored in a compact way,
        with shape `(..., M*(M+1)//2)`, where `...` is any number of
        leading batch dimensions.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., `[a00, a11, aa22, a01, a02, a12]`.
    vec : `(..., M) tensor`
        A vector, with shape `(..., M)`.
    eps : `float or (M,) sequence[float]`, optional
        Smoothing term added to the diagonal of `mat`

    Returns
    -------
    result : `(..., M) tensor`
        The solution of the linear system, with shape `(..., M)`.
    """

    # make the vector dimension first so that the code is less ugly
    backend = dict(dtype=mat.dtype, device=mat.device)
    mat = torch.movedim(mat, -1, 0)
    vec = torch.movedim(vec, -1, 0)
    nb_prm = len(vec)

    shape = torch.broadcast_shapes(mat.shape[1:], vec.shape[1:])
    shape = [vec.shape[0], *shape]

    diag = mat[:nb_prm]  # diagonal
    uppr = mat[nb_prm:]  # upper triangular part

    if eps is not None:
        # add smoothing term
        eps = torch.as_tensor(eps, **backend).flatten()
        eps = torch.cat([eps, eps[-1].expand(nb_prm - len(eps))])
        eps = eps.reshape([len(eps)] + [1] * (mat.dim() - 1))
        diag = diag + eps[:-1]

    if nb_prm == 1:
        res = vec / diag
    elif nb_prm == 2:
        res = _sym_solve2(diag, uppr, vec, shape)
    elif nb_prm == 3:
        res = _sym_solve3(diag, uppr, vec, shape)
    elif nb_prm == 4:
        res = _sym_solve4(diag, uppr, vec, shape)
    else:
        vec = torch.movedim(vec, 0, -1)
        mat = torch.movedim(mat, 0, -1)
        mat = sym_to_full(mat)
        return _solve_lu(mat, vec.unsqueeze(-1)).squeeze(-1)

    return torch.movedim(res, 0, -1)


def sym_det(mat: Tensor) -> Tensor:
    r"""Determinant of a compact symmetric matrix.

    Warning
    -------
    - Currently, autograd does not work through this function.

    Notes
    -----
    - Orders up to 4 are implemented in closed-form.
    - Orders > 4 use torch's batched implementation but require
      building the full matrices.
    - Backpropagation works at least for torch >= 1.6
      It should be checked on earlier versions.

    Parameters
    ----------
    mat : `(..., M*(M+1)//2) tensor`
        A $M \times M$  symmetric matrix that is stored in a compact way,
        with shape `(..., M*(M+1)//2)`, where `...` is any number of
        leading batch dimensions.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., `[a00, a11, aa22, a01, a02, a12]`.

    Returns
    -------
    result : `(...) tensor`
        Output determinant, with shape `(...)`.
    """

    # make the vector dimension first so that the code is less ugly
    mat = torch.movedim(mat, -1, 0)
    nb_prm = int((sqrt(1 + 8 * mat.shape[-1]) - 1) // 2)

    diag = mat[:nb_prm]  # diagonal
    uppr = mat[nb_prm:]  # upper triangular part

    if nb_prm == 1:
        res = diag
    elif nb_prm == 2:
        res = _sym_det2(diag, uppr)
    elif nb_prm == 3:
        res = _sym_det3(diag, uppr)
    elif nb_prm == 4:
        res = _sym_det4(diag, uppr)
    else:
        mat = torch.movedim(mat, 0, -1)
        mat = sym_to_full(mat)
        return torch.det(mat)

    return torch.movedim(res, 0, -1)


def sym_invert(mat: Tensor, diag: bool = False) -> Tensor:
    r"""Matrix inversion for compact symmetric matrices.

    Parameters
    ----------
    mat : `(..., M*(M+1)//2) tensor`
        A $M \times M$  symmetric matrix that is stored in a compact way,
        with shape `(..., M*(M+1)//2)`, where `...` is any number of
        leading batch dimensions.
        Its elements along the last (flat) dimension are the
        diagonal elements followed by the flattened upper-half elements.
        E.g., `[a00, a11, aa22, a01, a02, a12]`.
    diag : `bool`, default=False
        If True, only return the diagonal of the inverse

    Returns
    -------
    imat : `(..., M or M*(M+1)//2) tensor`
        Inverse matrix, with shape `(..., M or M*(M+1)//2)` (compact storage).

    """
    mat = torch.as_tensor(mat)
    nb_prm = int((sqrt(1 + 8 * mat.shape[-1]) - 1) // 2)
    if diag:
        imat = mat.new_empty([*mat.shape[:-1], nb_prm])
    else:
        imat = torch.empty_like(mat)

    cnt = nb_prm
    for i in range(nb_prm):
        e = mat.new_zeros(nb_prm)
        e[i] = 1
        vec = sym_solve(mat, e)
        imat[..., i] = vec[..., i]
        if not diag:
            for j in range(i+1, nb_prm):
                imat[..., cnt] = vec[..., j]
                cnt += 1
    return imat


def sym_outer(x: Tensor) -> Tensor:
    r"""Compute the symmetric outer product of a vector:
        $\mathbf{x} \times \mathbf{x}^T$

    Parameters
    ----------
    x : `(..., M) tensor`
        Input vector with shape `(..., M)`,
        where `...` is any number of leading batch dimensions.

    Returns
    -------
    xx : `(..., M*(M+1)//2) tensor`
        Output outer product with shape `(..., M*(M+1)//2)`.

    """
    M = x.shape[-1]
    MM = M*(M+1)//2
    xx = x.new_empty([*x.shape[:-1], MM])
    if x.requires_grad:
        xx[..., :M] = x.square()
        index = M
        for m in range(M):
            for n in range(m+1, M):
                xx[..., index] = x[..., m] * x[..., n]
    else:
        torch.mul(x, x, out=xx[..., :M])
        index = M
        for m in range(M):
            for n in range(m+1, M):
                torch.mul(x[..., m], x[..., n], out=xx[..., index])
                index += 1
    return xx


@torch.jit.script
def jhj1(jac, hess, out):
    # jac should be ordered as (D, D, ...)          [D=1]
    # hess should be ordered as (D*(D+1)//2, ...)   [D*(D+1)//2=1]
    # return ->  (D*(D+1)//2, ...)
    out[0] = jac[0, 0] * jac[0, 0] * hess[0]
    return out


@torch.jit.script
def jhj2(jac, hess, out):
    # jac should be ordered as (D, D, ...)          [D=2]
    # hess should be ordered as (D*(D+1)//2, ...)   [D*(D+1)//2=3]
    # return ->  (D*(D+1)//2, ...)
    #
    # Matlab symbolic toolbox:
    # out[00] = h00*j00^2 + h11*j01^2 + 2*h01*j00*j01
    # out[11] = h00*j10^2 + h11*j11^2 + 2*h01*j10*j11
    # out[01] = h00*j00*j10 + h11*j01*j11 + h01*(j01*j10 + j00*j11)
    h00 = hess[0]
    h11 = hess[1]
    h01 = hess[2]
    j00 = jac[0, 0]
    j01 = jac[0, 1]
    j10 = jac[1, 0]
    j11 = jac[1, 1]
    out[0] = j00 * j00 * h00 + j01 * j01 * h11 + 2 * j00 * j01 * h01
    out[1] = j10 * j10 * h00 + j11 * j11 * h11 + 2 * j10 * j11 * h01
    out[2] = j00 * j10 * h00 + j01 * j11 * h11 + (j01 * j10 + j00 * j11) * h01
    return out


@torch.jit.script
def jhj3(jac, hess, out):
    # jac should be ordered as (D, D, ...)          [D=3]
    # hess should be ordered as (D*(D+1)//2, ...)   [D*(D+1)//2=6]
    # return ->  (D*(D+1)//2, ...)
    #
    # Matlab symbolic toolbox:
    # out[00] = h00*j00^2 + 2*h01*j00*j01 + 2*h02*j00*j02 + h11*j01^2 + 2*h12*j01*j02 + h22*j02^2
    # out[11] = h00*j10^2 + 2*h01*j10*j11 + 2*h02*j10*j12 + h11*j11^2 + 2*h12*j11*j12 + h22*j12^2
    # out[22] = h00*j20^2 + 2*h01*j20*j21 + 2*h02*j20*j22 + h11*j21^2 + 2*h12*j21*j22 + h22*j22^2
    # out[01] = j10*(h00*j00 + h01*j01 + h02*j02) + j11*(h01*j00 + h11*j01 + h12*j02) + j12*(h02*j00 + h12*j01 + h22*j02)
    # out[02] = j20*(h00*j00 + h01*j01 + h02*j02) + j21*(h01*j00 + h11*j01 + h12*j02) + j22*(h02*j00 + h12*j01 + h22*j02)
    # out[12] = j20*(h00*j10 + h01*j11 + h02*j12) + j21*(h01*j10 + h11*j11 + h12*j12) + j22*(h02*j10 + h12*j11 + h22*j12)
    h00 = hess[0]
    h11 = hess[1]
    h22 = hess[2]
    h01 = hess[3]
    h02 = hess[4]
    h12 = hess[5]
    j00 = jac[0, 0]
    j01 = jac[0, 1]
    j02 = jac[0, 2]
    j10 = jac[1, 0]
    j11 = jac[1, 1]
    j12 = jac[1, 2]
    j20 = jac[2, 0]
    j21 = jac[2, 1]
    j22 = jac[2, 2]
    out[0] = h00 * j00 * j00 + 2 * h01 * j00 * j01 + 2 * h02 * j00 * j02 + h11 * j01 * j01 + 2 * h12 * j01 * j02 + h22 * j02 * j02
    out[1] = h00 * j10 * j10 + 2 * h01 * j10 * j11 + 2 * h02 * j10 * j12 + h11 * j11 * j11 + 2 * h12 * j11 * j12 + h22 * j12 * j12
    out[2] = h00 * j20 * j20 + 2 * h01 * j20 * j21 + 2 * h02 * j20 * j22 + h11 * j21 * j21 + 2 * h12 * j21 * j22 + h22 * j22 * j22
    out[3] = j10 * (h00 * j00 + h01 * j01 + h02 * j02) + j11 * (h01 * j00 + h11 * j01 + h12 * j02) + j12 * (h02 * j00 + h12 * j01 + h22 * j02)
    out[4] = j20 * (h00 * j00 + h01 * j01 + h02 * j02) + j21 * (h01 * j00 + h11 * j01 + h12 * j02) + j22 * (h02 * j00 + h12 * j01 + h22 * j02)
    out[5] = j20 * (h00 * j10 + h01 * j11 + h02 * j12) + j21 * (h01 * j10 + h11 * j11 + h12 * j12) + j22 * (h02 * j10 + h12 * j11 + h22 * j12)
    return out


@torch.jit.script
def jhjn(jac, hess, out):
    # jac should be ordered as (D, K, ...)
    # hess should be ordered as (K*(K+1)//2, ...)
    # return ->  (D*(D+1)//2, ...)

    K, D = jac.shape[:2]
    K2 = hess.shape[0]
    is_diag = K2 == K

    dacc = 0
    for d in range(D):
        doffset = (d + 1) * D - dacc  # offdiagonal offset
        dacc += d + 1
        # diagonal of output
        hacc = 0
        for k in range(K):
            hoffset = (k + 1) * K - hacc
            hacc += k + 1
            out[d] += hess[k] * jac[k, d].square()
            if not is_diag:
                for i, l in enumerate(range(k + 1, K)):
                    out[d] += 2 * hess[i + hoffset] * jac[k, d] * jac[l, d]
        # off diagonal of output
        for j, e in enumerate(range(d + 1, D)):
            hacc = 0
            for k in range(K):
                hoffset = (k + 1) * K - hacc
                hacc += k + 1
                out[j + doffset] += hess[k] * jac[k, d] * jac[k, e]
                if not is_diag:
                    for i, l in enumerate(range(k + 1, K)):
                        out[j + doffset] += hess[i + hoffset] * (
                                jac[k, d] * jac[l, e] + jac[l, d] * jac[k, e])
    return out


def sym_matmul(j, h):
    r"""
    Compute the symmetric matrix product
    $\mathbf{J}^\mathrm{T}\mathbf{H}\mathbf{J}$, where $\mathbf{H}$ is
    a compact symmetric matrix, and return a compact symmetric matrix.

    Parameters
    ----------
    j : (..., k, d) tensor
        Non symmetric matrix
    h : (..., k*(k+1)//2) tensor
        Symmetric matrix with compact storage.

    Returns
    -------
    jhj : (..., d*(d+1)//2) tensor
        Symmetric matrix with compact storage.
    """
    k, d = j.shape[-2:]
    h = h.movedim(-1, 0)
    j = j.movedim(-1, 0).movedim(-1, 0)
    d2 = d * (d + 1) // 2
    batch = torch.broadcast_shapes(j.shape[2:], h.shape[1:])
    out = h.new_zeros((d2,) + batch)
    if d == k == 1:
        out = jhj1(j, h, out)
    elif d == k == 2:
        out = jhj2(j, h, out)
    elif d == k == 3:
        out = jhj3(j, h, out)
    else:
        out = jhjn(j, h, out)
    out = out.movedim(0, -1)
    return out

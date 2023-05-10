__all__ = ['expm', 'logm', 'meanm']
import torch
from warnings import warn
from .sugar import lmdiv
from ._expm import expm, _expm
from ._logm import logm


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

"""

"""
__all__ = ['expm', 'logm', 'meanm', 'expm_derivatives']
import torch
from torch import Tensor
from warnings import warn
from .sugar import lmdiv
from ._impl.expm import expm, expm_derivatives
from ._impl.logm import logm


def meanm(mats: Tensor, max_iter: int = 1024, tol: float = 1e-20):
    r"""Compute the exponential barycentre of a set of matrices.

    Parameters
    ----------
    mats : (N, M, M) tensor
        Set of square invertible matrices
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for early stopping.
        The tolerance criterion is the sum-of-squares of the residuals
        in log-space, _i.e._,
        $\lVert\frac{1}{N}\sum_n \log_M(\mathbf{A}_n)\rVert_\mathrm{F}^2$

    Returns
    -------
    mean_mat : (M, M) tensor
        Mean matrix.

    References
    ----------
    1.  Pennec, X. and Arsigny, V., 2012.
        [**Exponential barycenters of the canonical Cartan connection and invariant means on Lie groups.**](https://hal.inria.fr/hal-00699361)
        In _Matrix information geometry_ (pp. 123-166).
        Berlin, Heidelberg: Springer Berlin Heidelberg.

            @incollection{pennec2012,
              title={Exponential barycenters of the canonical Cartan connection and invariant means on Lie groups},
              author={Pennec, Xavier and Arsigny, Vincent},
              booktitle={Matrix information geometry},
              pages={123--166},
              year={2012},
              publisher={Springer},
              url={https://hal.inria.fr/hal-00699361}
            }


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

    if not torch.is_tensor(mats):
        mats = torch.stack(mats)
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

__all__ = ['trapprox', 'vbald', 'maxeig_power']
import torch
from torch import Tensor
from typing import Callable, Union, Sequence, Literal, Optional
from math import ceil, log
from .sugar import lmdiv


def trapprox(
        matvec: Union[Tensor, Callable],
        shape: Optional[Sequence[int]] = None,
        moments: Optional[int] = None,
        samples: int = 10,
        method: Literal['rademacher', 'gaussian'] = 'rademacher',
        hutchpp: bool = False,
        **backend,
) -> Tensor:
    r"""Stochastic trace approximation (Hutchinson's estimator)

    Parameters
    ----------
    matvec : `sparse tensor or callable(tensor) -> tensor`
        Function that computes the matrix-vector product
    shape : `sequence[int]`
        "vector" shape
    moments : `int`, default=1
        Number of moments
    samples : `int`, default=10
        Number of samples
    method : `{'rademacher', 'gaussian'}`, default='rademacher'
        Sampling method
    hutchpp : `bool`, default=False
        Use Hutch++ instead of Hutchinson.
        /!\ Be aware that it uses more memory.

    Returns
    -------
    trace : `([moments],) tensor`

    Reference
    ---------
    1. Hutchinson, M.F., 1989.
    [**A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines.**]()
    _Communications in Statistics - Simulation and Computation_, 18(3), pp.1059-1076.

            @article{hutchinson1989,
              title={A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines},
              author={Hutchinson, Michael F},
              journal={Communications in Statistics-Simulation and Computation},
              volume={18},
              number={3},
              pages={1059--1076},
              year={1989},
              publisher={Taylor \& Francis}
            }

    2. Meyer, R.A., Musco, C., Musco, C. and Woodruff, D.P., 2021.
    [**Hutch++: Optimal stochastic trace estimation.**](https://arxiv.org/abs/2010.09649)
    _Symposium on Simplicity in Algorithms (SOSA)_ (pp. 142-155).
    Society for Industrial and Applied Mathematics.

            @inproceedings{meyer2021,
              title={Hutch++: Optimal stochastic trace estimation},
              author={Meyer, Raphael A and Musco, Cameron and Musco, Christopher and Woodruff, David P},
              booktitle={Symposium on Simplicity in Algorithms (SOSA)},
              pages={142--155},
              year={2021},
              organization={SIAM}
            }

    """
    if torch.is_tensor(matvec):
        mat = matvec
        matvec = lambda x: mat.matmul(x)
        backend.setdefault('dtype', mat.dtype)
        backend.setdefault('device', mat.device)
        shape = [*mat.shape[:-2], mat.shape[-1]]
    else:
        backend.setdefault('dtype', torch.get_default_dtype())
        backend.setdefault('device', torch.device('cpu'))
    no_moments = moments is None
    moments = moments or 1

    def rademacher(m=0):
        shape1 = [m, *shape] if m else shape
        x = torch.bernoulli(torch.full([], 0.5, **backend).expand(shape1))
        x.sub_(0.5).mul_(2)
        return x

    def gaussian(m=0):
        shape1 = [m, *shape] if m else shape
        return torch.randn(shape1, **backend)

    samp = rademacher if method[0].lower() == 'r' else gaussian

    if hutchpp:
        samples = int(ceil(samples/3))

        def matvecpp(x):
            y = torch.empty_like(x)
            for j in range(samples):
                y[j] = matvec(x[j])
            return y

        def dotpp(x, y):
            d = 0
            for j in range(samples):
                d += x[j].flatten().dot(y[j].flatten())
            return d

        def outerpp(x, y):
            z = x.new_empty([samples, samples])
            for j in range(samples):
                for k in range(samples):
                    z[j, k] = x[j].flatten().dot(y[k].flatten())
            return z

        def mmpp(x, y):
            z = torch.zeros_like(x)
            for j in range(samples):
                for k in range(samples):
                    z[j].addcmul_(x[k], y[k, j])
            return z

        t = torch.zeros([moments], **backend)
        q, g = samp(samples), samp(samples)
        q = torch.qr(matvecpp(q).T, some=True)[0].T
        g -= mmpp(q, outerpp(q, g))
        mq, mg = q, g
        for j in range(moments):
            mq = matvecpp(mq)
            mg = matvecpp(mg)
            t[j] = dotpp(q, mq) + dotpp(g, mg) / samples

    else:
        t = torch.zeros([moments], **backend)
        for i in range(samples):
            m = v = samp()
            for j in range(moments):
                m = matvec(m)
                t[j] += m.flatten().dot(v.flatten())
        t /= samples

    if no_moments:
        t = t[0]
    return t


def vbald(
        matvec: Union[Tensor, Callable],
        shape: Optional[Sequence[int]] = None,
        upper: Optional[float] = None,
        moments: int = 5,
        samples: int = 5,
        mc_samples: int = 64,
        method: Literal['rademacher', 'gaussian'] = 'rademacher',
        **backend,
) -> Tensor:
    """Variational Bayesian Approximation of Log Determinants

    Parameters
    ----------
    matvec : `sparse tensor or callable(tensor) -> tensor`
        Function that computes the matrix-vector product
    shape : `sequence[int]`
        "vector" shape
    upper : `float`
        Upper bound on eigenvalues
    moments : `int`, default=1
        Number of moments
    samples : `int`, default=5
        Number of samples for moment estimation
    mc_samples : `int`, default=64
        Number of samples for Monte Carlo integration
    method : `{'rademacher', 'gaussian'}`, default='rademacher'
        Sampling method

    Returns
    -------
    logdet : `scalar tensor`

    Reference
    ---------
    1. Granziol, D., Wagstaff, E., Ru, B.X., Osborne, M. and Roberts, S., 2018.
    [**VBALD-Variational Bayesian approximation of log determinants.**](https://arxiv.org/abs/1802.08054)
    _arXiv preprint_.

            @article{granziol2018,
              title={VBALD-Variational Bayesian approximation of log determinants},
              author={Granziol, Diego and Wagstaff, Edward and Ru, Bin Xin and Osborne, Michael and Roberts, Stephen},
              journal={arXiv preprint arXiv:1802.08054},
              year={2018}
            }

    """
    if torch.is_tensor(matvec):
        mat = matvec
        matvec = lambda x: mat.matmul(x)
        backend.setdefault('dtype', mat.dtype)
        backend.setdefault('device', mat.device)
        shape = [*mat.shape[:-2], mat.shape[-1]]
    else:
        backend.setdefault('dtype', torch.get_default_dtype())
        backend.setdefault('device', torch.device('cpu'))
    numel = torch.Size(shape).numel()

    if not upper:
        upper = maxeig_power(matvec, shape)
    matvec2 = lambda x: matvec(x).div_(upper)
    mom = trapprox(matvec2, shape, moments=moments, samples=samples,
                   method=method, **backend).cpu()
    mom /= numel

    # Compute beta parameters (Maximum Likelihood)
    alpha = mom[0] * (mom[0] - mom[1]) / (mom[1] - mom[0]**2)
    beta = alpha * (1/mom[0] - 1)
    if alpha > 0 and beta > 0:
        prior = torch.distributions.Beta(alpha.item(), beta.item())
    else:
        prior = torch.distributions.Uniform(1e-8, 1)

    # Compute coefficients
    coeff = _vbald_gn(mom, mc_samples, prior)

    # logdet(A) = N * (E[log(lam)] + log(upper))
    logdet = _vbald_mc_log(coeff, mc_samples, prior)
    logdet = numel * (logdet + log(upper))
    return logdet.to(backend['device'])


def _vbald_gn(mom, samples, prior, tol=1e-6, max_iter=512):
    dot = lambda u, v: u.flatten().dot(v.flatten())
    coeff = torch.zeros_like(mom)
    for n_iter in range(max_iter):
        loss, grad, hess = _vbald_mc(coeff, samples, prior,
                                     gradient=True, hessian=True)
        loss += dot(coeff, mom)
        grad = mom - grad
        diag = hess.diagonal(0, -1, -2)
        diag += 1e-3 * diag.abs().max() * torch.rand_like(diag)
        delta = lmdiv(hess, grad)

        success = False
        armijo = 1
        loss0 = loss
        coeff0 = coeff
        for n_iter in range(12):
            coeff = coeff0 - armijo * delta
            loss = _vbald_mc(coeff, samples, prior)
            loss += dot(coeff, mom)
            if loss < loss0:
                success = True
                break
            armijo /= 2
        if not success:
            return coeff0

        gain = abs(loss - loss0)
        if gain < tol:
            break
    return coeff


def _vbald_mc(coeff, samples, prior, gradient=False, hessian=False):
    nprm = 1
    if gradient:
        nprm += len(coeff)
    if hessian:
        nprm += len(coeff)

    # compute \int q(lam) * lam**j * exp(-1 - \sum coeff[i] lam**i) dlam
    # for multiple k, using monte carlo integration.
    s = coeff.new_zeros([nprm])
    for i in range(samples):
        lam = prior.sample([])
        q = _vbald_factexp(lam, coeff)
        s[0] += q
        if len(s) > 1:
            for j in range(1, len(s)):
                q = q * lam
                s[j] += q
    s /= samples

    # compute gradient and Hessian from the above integrals
    if gradient:
        g = s[1:len(coeff)+1]
        if hessian:
            h = g.new_zeros(len(coeff), len(coeff))
            for j in range(len(coeff)):
                for k in range(j+1, len(coeff)):
                    h[j, k] = h[k, j] = s[1+j+k]
                h[j, j] = s[1+j+j]
            return s[0], g, h
        return s[0], g
    return s[0]


def _vbald_factexp(lam, coeff):
    lam = lam ** torch.arange(1, len(coeff)+1, dtype=lam.dtype, device=lam.device)
    dot = lambda u, v: u.flatten().dot(v.flatten())
    return (-1 - dot(lam, coeff)).exp()


def _vbald_mc_log(coeff, samples, prior):

    # compute \int q(lam) * log(lam) * exp(-1 - \sum coeff[i] lam**i) dlam
    # for multiple k, using monte carlo integration.
    s = 0
    for i in range(samples):
        lam = prior.sample([])
        s += lam.log() * _vbald_factexp(lam, coeff)
    s /= samples
    return s


def maxeig_power(
        matvec: Union[Tensor, Callable],
        shape: Optional[Sequence[int]] = None,
        max_iter: int = 512,
        tol: float = 1e-6,
        **backend,
):
    """Estimate the maximum eigenvalue of a matrix by power iteration

    Parameters
    ----------
    matvec : `sparse tensor or callable(tensor) -> tensor`
        Function that computes the matrix-vector product
    shape : `sequence[int]`
        "vector" shape
    max_iter : `int`, default=512
    tol : `float`, default=1e-6

    Returns
    -------
    maxeig : `scalar tensor`
        Largest eigenvalue

    """

    if torch.is_tensor(matvec):
        mat = matvec
        matvec = lambda x: mat.matmul(x)
        backend.setdefault('dtype', mat.dtype)
        backend.setdefault('device', mat.device)
        shape = [*mat.shape[:-2], mat.shape[-1]]
    else:
        backend.setdefault('dtype', torch.get_default_dtype())
        backend.setdefault('device', torch.device('cpu'))

    dot = lambda u, v: u.flatten().dot(v.flatten())

    v = torch.bernoulli(torch.full([], 0.5, **backend).expand(shape)).sub_(0.5).mul_(2)
    mu = float('inf')

    for n_iter in range(max_iter):
        w, v = v, matvec(v)
        mu0, mu = mu, dot(w, v)
        v /= dot(v, v).sqrt_()
        if abs(mu - mu0) < tol:
            break
    return mu

__all__ = ['mvdigamma', 'besseli', 'besseli_ratio']
import torch
from typing import List
import math as pymath


@torch.jit.script
def mvdigamma(input, order: int = 1):
    """Derivative of the log of the Gamma function, eventually multivariate

    Parameters
    ----------
    input : tensor
        Input tensor
    order : int
        Multivariate order

    Returns
    -------
    output : tensor
        Output tensor
    """
    dg = torch.digamma(input)
    for p in range(2, order + 1):
        dg += torch.digamma(input + (1 - p) / 2)
    return dg


# Bessel functions were written by John Ashburner <john@fil.ion.ucl.ac.uk>
# (Wellcome Centre for Human Neuroimaging, UCL, London, UK)


def besseli(nu, z, mode=None):
    """Modified Bessel function of the first kind

    Parameters
    ----------
    nu : float
    z  : tensor
    mode : {0 or None, 1 or 'norm', 2 or 'log'}

    Returns
    -------
    b : tensor

        besseli(nu,z)        if mode is None
        besseli(ni,z)/exp(z) if mode == 'norm'
        log(besseli(nu,z))   if mode == 'log'

    References
    ----------
    1.  Garber, D.P., 1993.
        **On the use of the noncentral chi-square density function for
        the distribution of helicopter spectral estimates.**
        (No. NAS 1.26: 191546).
    """
    z = torch.as_tensor(z)
    is_scalar = z.dim() == 0
    if is_scalar:
        z = z[None]
    if not isinstance(mode, int):
        code = (2 if mode == 'log' else 1 if mode == 'norm' else 0)
    else:
        code = mode
    if nu == 0:
        z = _besseli0(z, code)
    elif nu == 1:
        z = _besseli1(z, code)
    else:
        z = _besseli_any(nu, z, code)
    if is_scalar:
        z = z[0]
    return z


@torch.jit.script
def _besseli0(z, code: int = 0):
    """Modified Bessel function of the first kind

    Parameters
    ----------
    z : `tensor`
    code : `{0, 1, 2}`

    Returns
    -------
    b : `tensor`
        ```
        besseli(0,z)        if code==0
        besseli(0,z)/exp(z) if code==1  ('norm')
        log(besseli(0,z))   if code==2  ('log')
        ```
    """
    f = torch.zeros_like(z)
    msk = z < 15.0/4.0

    # --- branch 1 ---
    if msk.any():
        zm = z[msk]
        t = (zm*(4.0/15.0))**2
        t = 1 + t*(3.5156229+t*(3.0899424+t*(1.2067492+t*(0.2659732+t*(0.0360768+t*0.0045813)))))
        if code == 2:
            f[msk] = t.log()
        else:
            if code == 1:
                t = t / zm.exp()
            f[msk] = t

    # --- branch 2 ---
    msk.bitwise_not_()
    if msk.any():
        zm = z[msk]
        t = (15.0/4.0)/zm
        t = (0.39894228+t*(0.01328592+t*(0.00225319+t*(-0.00157565+t*(0.00916281+t*(-0.02057706+t*(0.02635537+t*(-0.01647633+t*0.0039237))))))))
        t.clamp_min_(1e-32)
        if code == 2:
            f[msk] = zm - 0.5 * zm.log() + t.log()
        elif code == 1:
            f[msk] = t / zm.sqrt()
        else:
            f[msk] = zm.exp() * t / zm.sqrt()

    return f


@torch.jit.script
def _besseli1(z, code: int = 0):
    """Modified Bessel function of the first kind

    Parameters
    ----------
    z : tensor
    code : {0, 1, 2}

    Returns
    -------
    b : tensor
        besseli(1,z)        if code==0
        besseli(1,z)/exp(z) if code==1  ('norm')
        log(besseli(1,z))   if code==2  ('log')
    """
    f = torch.zeros_like(z)
    msk = z < 15.0/4.0

    # --- branch 1 ---
    if msk.any():
        zm = z[msk]
        t = (zm*(4.0/15.0))**2
        t = 0.5+t*(0.87890594+t*(0.51498869+t*(0.15084934+t*(0.02658733+t*(0.00301532+t*0.00032411)))))
        if code == 2:
            f[msk] = zm.log() + t.log()
        elif code == 0:
            f[msk] = zm * t
        else:
            f[msk] = zm * t / zm.exp()

    # --- branch 2 ---
    msk.bitwise_not_()
    if msk.any():
        zm = z[msk]
        t = (15.0/4.0)/zm
        t = 0.398942281+t*(-0.03988024+t*(-0.00362018+t*(0.00163801+t*(-0.01031555+t*(0.02282967+t*(-0.02895312+t*(0.01787654-t*0.00420059)))))))
        if code == 2:
            f[msk] = zm - 0.5 * zm.log() + t.log()
        elif code == 0:
            f[msk] = zm.exp() * t / zm.sqrt()
        else:
            f[msk] = t / zm.sqrt()
    return f


@torch.jit.script
def _besseli_small(nu: float, z, M: int = 64, code: int = 0):
    """Modified Bessel function of the first kind - series computation

    Parameters
    ----------
    nu : float
    z  : tensor
    M  : int, series length (bigger is more accurate, but slower)
    code : {0, 1, 2}

    Returns
    -------
    b : tensor
        besseli(nu,z)        if code==0
        besseli(nu,z)/exp(z) if code==1 ('norm')
        log(besseli(nu,z))   if code==2 ('log')
    """
    # The previous implementation of this function (`besseli_small_old`)
    # used `z` as a stabilizing pivot in the log-sum-exp. However,
    # this lead to underflows (`exp` returned zero). Instead, this new
    # implementation uses the first term in the sum (m=0) as pivot.
    # We therefore start with the second term and add 1 (= exp(0)) at the end.

    # NOTE: lgamma(3) = 0.693147180559945
    lgamma_nu_1 = pymath.lgamma(nu + 1)
    M = max(M, 2)
    x = torch.log(0.5*z)
    f = torch.exp(x * 2 - (0.693147180559945 + pymath.lgamma(nu + 2) - lgamma_nu_1))
    for m in range(2, M):
        f = f + torch.exp(x * (2*m) - (pymath.lgamma(m + 1) + pymath.lgamma(m + 1 + nu) - lgamma_nu_1))
    f = f + 1

    if code == 2:
        return f.log() + x * nu - lgamma_nu_1
    elif code == 1:
        return f * (x * nu - lgamma_nu_1 - z).exp()
    else:
        return f * (x * nu - lgamma_nu_1).exp()


# DEPRECATED
@torch.jit.script
def _besseli_small_old(nu: float, z, M: int = 64, code: int = 0):
    """Modified Bessel function of the first kind - series computation

    Parameters
    ----------
    nu : float
    z  : tensor
    M  : int, series length (bigger is more accurate, but slower)
    code : {0, 1, 2}

    Returns
    -------
    b : tensor
        besseli(nu,z)        if code==0
        besseli(nu,z)/exp(z) if code==1 ('norm')
        log(besseli(nu,z))   if code==2 ('log')
    """
    M = max(M, 2)
    x = torch.log(0.5*z)
    f = torch.exp(x * nu - pymath.lgamma(nu + 1.0) - z)
    for m in range(1, M):
        f = f + torch.exp(x * (2.0*m + nu) - (pymath.lgamma(m+1.0) + pymath.lgamma(m+1.0 + nu)) - z)

    if code == 2:
        return f.log() + z
    elif code == 1:
        return f
    else:
        return f * z.exp()


@torch.jit.script
def _besseli_large(nu: float, z, code: int = 0):
    """Modified Bessel function of the first kind

    Uniform asymptotic approximation (Abramowitz and Stegun p 378)

    Parameters
    ----------
    nu   : scalar float
    z    : torch tensor
    code : 0, 1 or 2

    Returns
    -------
    b : tensor
        besseli(nu,z)        if code==0
        besseli(nu,z)/exp(z) if code==1 ('norm')
        log(besseli(nu,z))   if code==2 ('log')
    """

    f = z/nu
    f = f*f
    msk = f > 4.0
    t = torch.zeros_like(f)

    # -- branch 1 ---
    if msk.any():
        tmp = torch.sqrt(1.0+f[msk].reciprocal())
        t[msk] = z[msk]*tmp/nu
        f[msk] = nu*(t[msk] + torch.log((nu/z[msk]+tmp).reciprocal()))
    # -- branch 2 ---
    msk = msk.bitwise_not_()
    if msk.any():
        tmp = torch.sqrt(1.0+f[msk])
        t[msk] = tmp.clamp_max(1)
        f[msk] = nu*(t[msk] + torch.log(z[msk]/(nu*(1.0+tmp))))

    t = t.reciprocal()
    tt = t*t
    ttt = t*tt
    us = 1.0
    den = nu
    us = us + t*(0.125 - tt*0.2083333333333333)/den
    den = den*nu
    us = us + tt*(0.0703125 + tt*(-0.4010416666666667 + tt*0.3342013888888889))/den
    den = den*nu
    us = us + ttt*(0.0732421875 + tt*(-0.8912109375 + tt*(1.846462673611111 - tt*1.025812596450617)))/den
    den = den*nu
    us = us + tt*tt*(0.112152099609375 + tt*(-2.3640869140625 + tt*(8.78912353515625 +
               tt*(-11.20700261622299 + tt*4.669584423426248))))/den
    den = den*nu
    us = us + tt*ttt*(0.2271080017089844 + tt*(-7.368794359479632 + tt*(42.53499874638846 +
               tt*(-91.81824154324002 + tt*(84.63621767460074 - tt*28.21207255820025)))))/den
    den = den*nu
    us = us + ttt*ttt*(0.5725014209747314 + tt*(-26.49143048695155 + tt*(218.1905117442116 +
               tt*(-699.5796273761326 + tt*(1059.990452528 + tt*(-765.2524681411817 +
               tt*212.5701300392171))))))/den

    if code == 2:
        f = f + 0.5*(torch.log(t)-pymath.log(nu)) - 0.918938533204673 # 0.5*log(2*pi)
        f = f + torch.log(us)
    elif code == 0:
        f = torch.exp(f)*torch.sqrt(t)*us*(0.398942280401433/pymath.sqrt(nu))
    else:
        f = torch.exp(f-z)*torch.sqrt(t)*us*(0.398942280401433/pymath.sqrt(nu))
    return f


@torch.jit.script
def _besseli_any(nu: float, z, code: int = 0):
    """
    Modified Bessel function of the first kind

    Parameters
    ----------
    nu : float
    z  : tensor
    code : {0, 1, 2}

    Returns
    -------
    b : tensor
        besseli(nu,z)        if code==0
        besseli(nu,z)/exp(z) if code==1 ('norm')
        log(besseli(nu,z))   if code==2 ('log')

    """

    if nu >= 15.0:
        f = _besseli_large(nu, z, code)
    else:
        thr = 5.0 * pymath.sqrt(15.0 - nu) * pymath.sqrt(nu + 15.0) / 3.0
        msk = z < 2.0 * thr
        f = torch.zeros_like(z)
        if msk.any():
            f[msk] = _besseli_small(nu, z[msk], int(pymath.ceil(thr * 1.9 + 2.0)), code)
        msk = msk.bitwise_not_()
        if msk.any():
            f[msk] = _besseli_large(nu, z[msk], code)
    return f


@torch.jit.script
def besseli_ratio(nu: float, X, N: int = 4, K: int = 10):
    """Approximates ratio of the modified Bessel functions of the first kind:
       besseli(nu+1,x)/besseli(nu,x)

    Parameters
    ----------
    nu : float
    X : tensor
    N, K :  int
        Terms in summation, higher number, better approximation.

    Returns
    -------
    I : tensor
        Ratio of the modified Bessel functions of the first kind.

    References
    ----------
    1. Amos, D.E., 1974.
      **Computation of modified Bessel functions and their ratios.**
      _Mathematics of computation_, 28(125), pp.239-251.
    """

    # Begin by computing besseli(nu+1+N,x)/besseli(nu+N,x)

    nu1 = nu + K
    rk: List[torch.Tensor] = []
    XX = X*X

    # Lower bound (eq. 20a)
    # for k=0:N, rk{k+1} = x./((nu1+k+0.5)+sqrt((nu1+k+1.5).^2+x.^2)); end
    for k in range(0, N + 1):
        tmp = XX + (nu1 + k + 1.5) ** 2
        tmp = tmp.sqrt()
        tmp = tmp + (nu1 + k + 0.5)
        tmp = X / tmp
        rk.append(tmp)

    for m in range(N, 0, -1):
        # Recursive updates (eq. 20b)
        # rk{k} = x./(nu1+k+sqrt((nu1+k).^2+((rk{k+1})./rk{k}).
        for k2 in range(1, m + 1):
            tmp = rk[k2] / rk[k2-1]
            tmp = tmp * XX
            tmp = tmp + (nu1 + k2) ** 2
            tmp = tmp.sqrt()
            tmp = tmp + (nu1 + k2)
            tmp = X / tmp
            rk[k2-1] = tmp
        rk.pop(-1)
    result = rk.pop(0)

    # Convert the result to besseli(nu+1,x)/besseli(nu,x) with
    # backward recursion (eq. 2).
    iX = X.reciprocal()
    for k3 in range(K, 0, -1):
        # result = 1. / (2. * (nu + k3) / X + result)
        result = result.add(iX, alpha=2 * (nu + k3))
        result = result.reciprocal()

    return result

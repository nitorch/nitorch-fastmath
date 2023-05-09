import torch


def round(t, decimals=0):
    """Round a tensor to the given number of decimals.

    Parameters
    ----------
    t : `tensor`
        Input tensor.
    decimals : `int`, default=0
        Round to this decimal.

    Returns
    -------
    t : `tensor`
        Rounded tensor.
    """
    return torch.round(t * 10 ** decimals) / (10 ** decimals)

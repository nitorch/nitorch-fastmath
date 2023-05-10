"""
## Overview
This module contains linear-algebra routines (matrix-vector product,
matrix inversion, etc.) for batches of symmetric matrices stored in a
compact way (that is, only $N(N+1)/2$ values are stored, instead of $N^2$).

Our compact representation differs from classical "columns" or "rows"
layouts. The compact flattened matrix should contain the diagonal of
the matrix first, followed by the rows of the upper triangle of the
matrix, i.e.:

    [ a d e ]
    [ . b f ]   =>  [a b c d e f]
    [ . . c ]

Note that matrix-vector functions (`matvec`, `solve`) also accept (and
automatically detect) compact diagonal matrices and compact scaled identity
matrices. If the vector has shape `(*, N)` and the matrix has shape `(*, NN)`,
where `*` is any number of leading batch dimensions, `NN` can take values:

- `1`: and the matrix is assumed to be a scaled identity,
- `N`: and the matrix is assumed to be a diagonal matrix,
- `N*(N+1)//2`: and the matrix is assumed to be symmetric,
- `N*N`: and the matrix is assumed to be full.

---
"""
__all__ = [
    'sym_to_full', 'sym_diag', 'sym_outer', 'sym_det', 'sym_matmul',
    'sym_matvec',
    'sym_addmatvec', 'sym_addmatvec_',
    'sym_submatvec', 'sym_submatvec_',
    'sym_solve', 'sym_solve_',
    'sym_invert', 'sym_invert_'
]
from ._impl.sym import sym_to_full, sym_diag, sym_outer, sym_det, sym_matmul
from jitfields.sym import *

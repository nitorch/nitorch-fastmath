"""
## Overview

I found that some torch functions (e.g., `inverse()` or `det()`) were
not so efficient when applied to large batches of small matrices,
especially on the GPU (this is not so true on the CPU). I reimplemented
them using torchscript for 2x2 and 3x3 matrices, and they are much
faster.

I used to have a `batchmatmul` too, but its speed was not always better
than `torch.matmul()` (it depended a lot on the striding layout),
so I removed it.

---
"""
__all__ = ['batchmatvec', 'batchdet', 'batchinv']
from ._impl.batched import batchmatvec, batchdet, batchinv

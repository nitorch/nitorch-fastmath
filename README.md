# nitorch-fastmath
Math and linear algebra routines that are (sometimes) faster than vanilla PyTorch

## Overview

This package implements an array of math routines in PyTorch.
They serve multiple purposes:

- **Compact symmetric matrices:** Symmetric matrices can be stored using a
  compact layout that only contains the upper (or lower) half of the matrix.
  We expose fast functions for such compact symmetric matrices that were
  implemented in [`jitfields`](https://github.com/balbasty/jitfields).
- **Fast batched math:** Many linear-algebra routines in PyTorch are optimized
  for large matrices, but not for large batches of small matrices. We use
  TorchScript to implement faster routines for these such batched matrices.
- **Reduction that omit NaNs:** we reimplement reduction functions (sum,
  mean, min, max, _etc._) that omit NaNs (instead of returning NaNs, as
  is the case in PyTorch).
- **Special functions:** Many special functions were only implemented in
  recent versions of PyTorch. Some of them are even still missing. We
  provide TorchScript implementations for some if these special functions,
  that are compatible with PyTorch versions as old as `1.8`.
- **Matrix exponential and logarithm:** We use cupy to expose matrix functions
  that are not available in old versions of PyTorch.

In general, there is a risk that these routines are less precise and
not as well tested as their `pytorch` or `scipy` equivalents.
If precision matters to you, stick to these reference implementations.

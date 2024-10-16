from .base import DenseTransform, InverseDenseTransform
from .dct import _dct_kernels


class DenseDCT(DenseTransform):
    def __init__(self, size: int = 3):
        super().__init__(kernels=_dct_kernels(size))


class DenseIDCT(InverseDenseTransform):
    def __init__(self, size: int = 3):
        super().__init__(kernels=_dct_kernels(size))

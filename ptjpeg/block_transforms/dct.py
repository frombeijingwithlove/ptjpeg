import math

import torch

from .base import BlockTransform, InverseBlockTransform


def _dct_kernels(size):
    c_dc = math.sqrt(1 / size)
    c_ac = math.sqrt(2 / size)
    kernels = torch.empty(size * size, size, size)
    x, y = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='xy')
    with torch.no_grad():
        for i in range(0, size * size):
            ix = i % size
            iy = i // size
            kernels[i] = (c_dc if ix == 0 else c_ac) * \
                         (c_dc if iy == 0 else c_ac) * \
                         torch.cos((2 * x + 1) * ix * torch.pi / (2 * size)) * \
                         torch.cos((2 * y + 1) * iy * torch.pi / (2 * size))

    return kernels


class DCT(BlockTransform):
    def __init__(self, size: int = 8):
        super().__init__(kernels=_dct_kernels(size))


class IDCT(InverseBlockTransform):
    def __init__(self, size: int = 8):
        super().__init__(kernels=_dct_kernels(size))

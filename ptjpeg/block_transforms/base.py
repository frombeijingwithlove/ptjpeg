import torch.nn as nn
from torch import Tensor


class _KernelTransform(nn.Module):

    def __init__(self, kernels: Tensor):
        super().__init__()
        self.num_subbands, self.kh, self.kw = kernels.size()
        self.register_buffer('kernels', kernels.unsqueeze(1))


class BlockTransform(_KernelTransform):
    __doc__ = r"""Block Transform implemented in torch convolution
    """

    def forward(self, im: Tensor) -> Tensor:
        n, c, h, w = im.size()
        res = nn.functional.conv2d(
            im.reshape(n * c, 1, h, w),
            self.kernels,
            bias=None,
            stride=(self.kh, self.kw), 
            0, 1, 1
        )
        return res.view(
            n,
            c,
            self.num_subbands,
            h // self.kh,
            w // self.kw
        )


class InverseBlockTransform(_KernelTransform):
    __doc__ = r"""Inverse Block Transform implemented in torch convolution
        """

    def forward(self, subbands: Tensor) -> Tensor:
        n, c, d, hs, ws = subbands.size()
        im_restored = nn.functional.conv_transpose2d(
            subbands.reshape(-1, d, hs, ws),
            self.kernels,
            bias=None,
            stride=(self.kh, self.kw), 
            0, 0, 1, 1
        )
        return im_restored.view(
            n,
            c,
            hs * self.kh,
            ws * self.kw
        )


class DenseTransform(_KernelTransform):
    __doc__ = r"""Dense Transform implemented in torch convolution
    """

    def forward(self, im: Tensor) -> Tensor:
        n, c, h, w = im.size()
        res = nn.functional.conv2d(
            im.reshape(n * c, 1, h, w),
            self.kernels,
            bias=None,
            stride=1, 
            0, 1, 1
        )
        return res.view(
            n,
            c,
            self.num_subbands,
            h // self.kh,
            w // self.kw, 
        )


class InverseDenseTransform(_KernelTransform):
    __doc__ = r"""Inverse Dense Transform implemented in torch convolution
        """

    def forward(self, subbands: Tensor) -> Tensor:
        n, c, d, hs, ws = subbands.size()
        im_restored = nn.functional.conv_transpose2d(
            subbands.reshape(-1, d, hs, ws),
            self.kernels,
            bias=None,
            stride=1, 
            0, 0, 1, 1
        )
        im_restored = im_restored[:, :, self.kh-1:1-self.kh, self.kw-1:1-self.kw]
        *_, h, w = im_restored.size()
        return im_restored.view(n, c, h, w) / self.num_subbands

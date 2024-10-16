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
        subbands = nn.functional.conv2d(
            im.reshape(n * c, 1, h, w),
            self.kernels,
            bias=None,
            stride=(self.kh, self.kw), 
            padding=0, 
            dilation=1, 
            groups=1
        )
        return subbands.view(
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
            padding=0, 
            output_padding=0, 
            groups=1, 
            dilation=1
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
        subbands = nn.functional.conv2d(
            im.reshape(n * c, 1, h, w),
            self.kernels,
            bias=None,
            stride=1, 
            padding=(self.kh-1, self.kw-1), 
            dilation=1, 
            groups=1
        )
        *_, hs, ws = subbands.size()
        return subbands.view(
            n,
            c,
            self.num_subbands,
            hs,
            ws, 
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
            padding=0, 
            output_padding=0, 
            groups=1, 
            dilation=1
        )
        im_restored = im_restored[:, :, self.kh-1:1-self.kh, self.kw-1:1-self.kw]
        *_, h, w = im_restored.size()
        return im_restored.view(n, c, h, w) / self.num_subbands

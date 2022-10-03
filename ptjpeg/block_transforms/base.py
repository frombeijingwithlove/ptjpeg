import torch.nn as nn
from torch import Tensor


class _BlockTransformKernels(nn.Module):

    def __init__(self, kernels: Tensor):
        super().__init__()
        self.num_subbands, self.kh, self.kw = kernels.size()
        self.register_buffer('kernels', kernels.unsqueeze(1))


class BlockTransform(_BlockTransformKernels):
    __doc__ = r"""Block Discrete Transform (DCT) implemented in torch convolution
    """

    def forward(self, im: Tensor) -> Tensor:
        n, c, h, w = im.size()
        res = nn.functional.conv2d(
            im.reshape(n * c, 1, h, w),
            self.kernels,
            bias=None,
            stride=(self.kh, self.kw),
        )
        return res.view(
            n,
            c,
            self.num_subbands,
            h // self.kh,
            w // self.kw
        )


class InverseBlockTransform(_BlockTransformKernels):
    __doc__ = r"""Block Discrete Transform implemented in torch convolution
        """

    def forward(self, subbands: Tensor) -> Tensor:
        n, c, d, hs, ws = subbands.size()
        im_restored = nn.functional.conv_transpose2d(
            subbands.reshape(-1, d, hs, ws),
            self.kernels,
            bias=None,
            stride=(self.kh, self.kw),
        )
        return im_restored.view(
            n,
            c,
            hs * self.kh,
            ws * self.kw
        )

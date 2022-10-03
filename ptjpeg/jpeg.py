import torch.nn as nn
from torch import Tensor
import torch

from ptjpeg import DCT, IDCT, RGB2YCbCr, YCbCr2RGB, StraightThroughQuantizer, QTableScaler
from ptjpeg.utils import check_image_tensor


class PTJPEG(nn.Module):
    def __init__(self, chroma_sampling='h2v2', max_size=2048):
        assert chroma_sampling in ('h1v1', 'h2v1', 'h2v2')

        super().__init__()

        self.rgb2ycbcr = RGB2YCbCr()
        self.dct = DCT()
        self.scaler = QTableScaler()
        self.quantizer = StraightThroughQuantizer()
        self.idct = IDCT()
        self.ycbcr2rgb = YCbCr2RGB()

        self.register_buffer(
            'chroma_downsample_dither', (torch.arange(max_size) % 2 + 1).view(1, 1, 1, -1).float()
        )
        if chroma_sampling == 'h2v2':
            self.downsample_chroma = self._chroma_downsample_h2v2
            self.upsample_chroma = self._chroma_upsample_h2v2
        elif chroma_sampling == 'h2v1':
            self.downsample_chroma = self._chroma_downsample_h2v1
            self.upsample_chroma = self._chroma_upsample_h2v1
        else:
            self.downsample_chroma = nn.Identity()
            self.upsample_chroma = nn.Identity()

    def encode(self, im_tensor: Tensor, q: int = 75):
        pass

    def decode(self, im_path: str) -> Tensor:
        pass

    def forward(self, im_tensor: Tensor, quality: int = 75) -> Tensor:
        check_image_tensor(im_tensor)
        return self._forward_rgb(im_tensor * 255., quality) / 255.

    def _forward_rgb(self, im_tensors: Tensor, quality: int, chroma_sampling='2x2') -> Tensor:
        # RGB --> YCbCr
        ycbcr = self.rgb2ycbcr(im_tensors) - 128
        y = ycbcr[:, :1]
        cbcr = ycbcr[:, 1:]
        cbcr = self.downsample_chroma(cbcr)

        import numpy
        cb_ref = numpy.genfromtxt('aux_res/ttt_cb_ds.csv', delimiter=',')
        cr_ref = numpy.genfromtxt('aux_res/ttt_cr_ds.csv', delimiter=',')

        cb, cr = cbcr.chunk(2, dim=1)

        cb_error = cb + 128 - cb_ref
        cr_error = cr + 128 - cr_ref

        downsampling_bias = 1   # 121212... dither pattern

        y_subbands = self.dct(y)
        cbcr_subbands = self.dct(cbcr)

        luma_qtable, chroma_qtable = self.scaler(quality)

        y_subbands_q = self.quantizer(y_subbands / luma_qtable) * luma_qtable
        cbcr_subbands_q = self.quantizer(cbcr_subbands / chroma_qtable) * chroma_qtable

        y_q = self.idct(y_subbands_q)

        cbcr_q = self.idct(cbcr_subbands_q)
        cbcr_q = self.upsample_chroma(cbcr_q)

        ycbcr_q = torch.cat([y_q, cbcr_q], dim=1) + 128
        return self.ycbcr2rgb(ycbcr_q).clamp_(0, 255)

    def _chroma_downsample_h2v1(self, cbcr):
        raise NotImplementedError('h2v1 is not implemented ...')

    def _chroma_downsample_h2v2(self, cbcr):
        cbcr = nn.functional.interpolate(cbcr, None, scale_factor=0.5, mode='bilinear')
        cbcr = cbcr + self.chroma_downsample_dither[..., :cbcr.size(-1)] / 4
        return cbcr + (cbcr.floor() - cbcr).detach()

    def _chroma_upsample_h2v1(self, cbcr):
        raise NotImplementedError('h2v1 is not implemented ...')

    def _chroma_upsample_h2v2(self, cbcr):
        cbcr = nn.functional.interpolate(cbcr, None, scale_factor=2, mode='nearest')
        return cbcr

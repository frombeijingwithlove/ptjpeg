"""Following jcparam.c:
jpeg_quality_scaling (int quality)
/* Convert a user-specified quality rating to a percentage scaling factor
 * for an underlying quantization table, using our recommended scaling curve.
 * The input 'quality' factor should be 0 (terrible) to 100 (very good).
 */
{
  /* Safety limit on quality factor.  Convert 0 to 1 to avoid zero divide. */
  if (quality <= 0) quality = 1;
  if (quality > 100) quality = 100;

  /* The basic table is used as-is (scaling 100) for a quality of 50.
   * Qualities 50..100 are converted to scaling percentage 200 - 2*Q;
   * note that at Q=100 the scaling is 0, which will cause jpeg_add_quant_table
   * to make all the table entries 1 (hence, minimum quantization loss).
   * Qualities 1..50 are converted to scaling percentage 5000/Q.
   */
  if (quality < 50)
    quality = 5000 / quality;
  else
    quality = 200 - quality*2;

  return quality;
}
"""

import torch
from torch import Tensor
import torch.nn as nn
from ._qtable import LUMINANCE_QTABLE, CHROMINANCE_QTABLE


def _jpeg_quality_scaling(quality: int = 75):
    quality = max(1, min(100, quality))
    return 5000 // quality if quality < 50 else 200 - quality * 2


class QTableScaler(nn.Module):
    def __init__(self,
                 luma_qtable=LUMINANCE_QTABLE,
                 chroma_qtable=CHROMINANCE_QTABLE
                 ):
        super().__init__()
        self.register_buffer(
            'luma_qtable',
            torch.LongTensor(luma_qtable).view(1, len(luma_qtable), 1, 1)
        )
        self.register_buffer(
            'chroma_qtable',
            torch.LongTensor(chroma_qtable).view(1, len(chroma_qtable), 1, 1)
        )

    @staticmethod
    def _scale_qtable(qtable: Tensor, scale_factor: int) -> Tensor:
        scaled_qtable = torch.div(qtable * scale_factor + 50, 100, rounding_mode='floor')
        return scaled_qtable.clamp_(1, 255).float().detach()

    def forward(self, quality: int) -> tuple[Tensor, Tensor]:
        scale_factor = _jpeg_quality_scaling(quality)
        luma_qtable = self._scale_qtable(self.luma_qtable, scale_factor)
        chroma_qtable = self._scale_qtable(self.chroma_qtable, scale_factor)
        return luma_qtable, chroma_qtable

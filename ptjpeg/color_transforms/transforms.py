"""From libjpeg jcolor.c

 *	Y  =  0.299 * R + 0.587 * G + 0.114 * B
 *	Cb = -0.168735892 * R - 0.331264108 * G + 0.5 * B + CENTERJSAMPLE
 *	Cr =  0.5 * R - 0.418687589 * G - 0.081312411 * B + CENTERJSAMPLE

    rgb_ycc_tab[i+R_Y_OFF] = FIX(0.299) * i;
    rgb_ycc_tab[i+G_Y_OFF] = FIX(0.587) * i;
    rgb_ycc_tab[i+B_Y_OFF] = FIX(0.114) * i   + ONE_HALF;

    rgb_ycc_tab[i+R_CB_OFF] = (- FIX(0.168735892)) * i;
    rgb_ycc_tab[i+G_CB_OFF] = (- FIX(0.331264108)) * i;
    rgb_ycc_tab[i+B_CB_OFF] = FIX(0.5) * i    + CBCR_OFFSET + ONE_HALF-1;

    rgb_ycc_tab[i+G_CR_OFF] = (- FIX(0.418687589)) * i;
    rgb_ycc_tab[i+B_CR_OFF] = (- FIX(0.081312411)) * i;

"""

import torch
import torch.nn as nn


from ._standards import COEFFICIENTS

_SCALE = 65536.
_ONE_HALF = 32768.
_MAXJSAMPLE = 255.
_CENTERJSAMPLE = 128.


def _float_cast(w, p):
    z = 10 ** p
    return [round(_ * z) / z for _ in w]


class _ColorTransformBase(nn.Module):
    def __init__(self, weight, bias, precision_cast=9):
        super().__init__()
        if precision_cast > 0:
            weight = _float_cast(weight, precision_cast)
            bias = _float_cast(bias, precision_cast)
        weight = torch.Tensor(weight)
        bias = torch.Tensor(bias)
        weight = (weight * _SCALE).round()
        bias = (bias * _SCALE).round()
        bias[0] += _ONE_HALF
        bias[1:] += _ONE_HALF - 1
        self.register_buffer('weight', weight.view(3, 3, 1, 1))
        self.register_buffer('bias', bias)

    def forward(self, batch):
        output = nn.functional.conv2d(batch, self.weight, bias=self.bias) / _SCALE
        output_approx = output + (torch.floor(output) - output).detach()

        #DEBUG write to csv
        #import numpy
        #y_ref = numpy.genfromtxt('aux_res/ttt_y.csv', delimiter=',')
        #cb_ref = numpy.genfromtxt('aux_res/ttt_cb.csv', delimiter=',')
        #cr_ref = numpy.genfromtxt('aux_res/ttt_cr.csv', delimiter=',')

        #y_error = output_approx[0][0] - y_ref
        #cb_error = output_approx[0][1] - cb_ref
        #cr_error = output_approx[0][2] - cr_ref

        return output_approx


class RGB2YCbCr(_ColorTransformBase):
    def __init__(self, std_name='BT.601'):
        Kr, Kg, Kb = COEFFICIENTS[std_name]
        weight = [
            Kr, Kg, Kb,
            -Kr / 2 / (1 - Kb), -Kg / 2 / (1 - Kb), 1 / 2,
            1 / 2, -Kg / 2 / (1 - Kr), -Kb / 2 / (1 - Kr)
        ]
        bias = (0, _CENTERJSAMPLE, _CENTERJSAMPLE)
        super().__init__(weight, bias)

#TODO
class YCbCr2RGB(_ColorTransformBase):
    def __init__(self, std_name='BT.601'):
        Kr, Kg, Kb = COEFFICIENTS[std_name]
        weight = [
            1, 0, 2 * (1 - Kr),
            1, 2 * (Kb - 1) * Kb / Kg, 2 * (Kr - 1) * Kr / Kg,
            1, 2 * (1 - Kb), 0
        ]
        bias = (
            (Kr - 1) * _MAXJSAMPLE,
            ((1 - Kb) * Kb / Kg + (1 - Kr) * Kr / Kg) * _MAXJSAMPLE,
            (Kb - 1) * _MAXJSAMPLE
        )
        super().__init__(weight, bias)

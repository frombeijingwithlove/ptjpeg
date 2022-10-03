import torch
import torch.nn as nn
from torch import Tensor


class _QuantizerBase(nn.Module):

    def _training_forward(self, input_tensor: Tensor) -> Tensor:
        raise NotImplementedError()

    @staticmethod
    def _inference_forward(input_tensor: Tensor) -> Tensor:
        return input_tensor.round()

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self._training_forward(input_tensor) if self.training \
            else self._inference_forward(input_tensor)


class StraightThroughQuantizer(_QuantizerBase):
    """
    Tensorflow API: tf.stopgradient??
    """
    def _training_forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor + (input_tensor.round() - input_tensor).detach()


class UniformSamplingQuantizer(_QuantizerBase):
    """
    End-to-End Optimization of Nonlinear Transform Codes for Perceptual Quality
    """
    def _training_forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor + torch.empty_like(input_tensor).uniform_(-0.5, 0.5)


class CubicApproxQuantizer(_QuantizerBase):
    """
    JPEG-resistant Adversarial Images
    """
    def _training_forward(self, input_tensor: Tensor) -> Tensor:
        q_error = input_tensor.round() - input_tensor
        return input_tensor + q_error.detach() - torch.pow(q_error, 3)

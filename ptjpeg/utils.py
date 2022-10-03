from torch import Tensor


def check_image_tensor(input_tensor: Tensor) -> None:
    if not input_tensor.is_floating_point() \
            or len(input_tensor.size()) != 4 \
            or input_tensor.size(1) not in (1, 3) \
            or input_tensor.max() > 255 \
            or input_tensor.min() < 0:
        raise ValueError('Expected an image tensor with Nx3xHxW or Nx1xHxW.')

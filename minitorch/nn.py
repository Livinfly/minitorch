# flake8: noqa
from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor
import numpy as np


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    input = input.contiguous()
    input = input.view(batch, channel, new_height, kh, new_width, kw)
    input = input.permute(0, 1, 2, 4, 3, 5)
    input = input.contiguous()
    return (
        input.view(batch, channel, new_height, new_width, kh * kw),
        new_height,
        new_width,
    )
    raise NotImplementedError("Need to implement for Task 4.3")


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    tile_tensor, new_height, new_width = tile(input, kernel)
    avgpool_tensor = tile_tensor.mean(4)
    return avgpool_tensor.view(batch, channel, new_height, new_width)
    raise NotImplementedError("Need to implement for Task 4.3")


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        dims = dim.to_numpy().astype(int).tolist()
        max_tensor = max_reduce(input, dims[0])
        ctx.save_for_backward(input, max_tensor)
        return max_tensor
        raise NotImplementedError("Need to implement for Task 4.4")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        input, max_tensor = ctx.saved_values
        return grad_output * (input == max_tensor), 0.0
        raise NotImplementedError("Need to implement for Task 4.4")


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    # TODO: Implement for Task 4.4.
    max_tensor = max(input, dim)
    input = input - max_tensor
    input = input.exp()
    return input / input.sum(dim)
    raise NotImplementedError("Need to implement for Task 4.4")


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    # TODO: Implement for Task 4.4.
    max_tensor = max(input, dim)
    shift_tensor = input - max_tensor
    exp_tensor = shift_tensor.exp()
    sum_tensor = exp_tensor.sum(dim)
    log_tensor = sum_tensor.log()
    return shift_tensor - log_tensor
    raise NotImplementedError("Need to implement for Task 4.4")


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    tiled_tensor, new_height, new_width = tile(input, kernel)
    max_pooled_tensor = max(tiled_tensor, 4)
    return max_pooled_tensor.view(batch, channel, new_height, new_width)
    raise NotImplementedError("Need to implement for Task 4.4")


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    # TODO: Implement for Task 4.4.
    if ignore:
        return input
    mask = rand(input.shape) > rate
    dropout_tensor = input * mask
    return dropout_tensor
    raise NotImplementedError("Need to implement for Task 4.4")

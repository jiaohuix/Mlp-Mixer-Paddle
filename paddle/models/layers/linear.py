""" Linear layer (alternate definition)
"""
import paddle
import paddle.nn.functional as F
from paddle import nn as nn
import torch

class Linear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Wraps paddle.nn.Linear to support AMP + paddlescript usage by manually casting
    weight & bias to input.dtype to work around an issue w/ paddle.addmm in this use case.
    """
    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        # if paddle.jit.is_scripting():
        #     bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
        #     return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        # else:
        return F.linear(input, self.weight, self.bias)

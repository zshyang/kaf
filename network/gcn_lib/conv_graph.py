from typing import Union

import add_sys_path
import torch.nn as nn
from gcn_lib.ffn import FFN
from gcn_lib.grapher import Grapher


class ConvGraph(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: str = 'relu',
        bias: bool = True,
        conv: str = 'edge',
        dilation: int = 1,
        dimension: int = 3,
        drop_path: float = 0.0,
        epsilon: float = 0.0,
        kernel_size: int = 9,
        n: int = 8,
        norm: Union[None, str] = None,
        pick_y_method: str = 'avg_pool',
        pick_y_number: int = 100,
        r: int = 2,
        relative_pos: bool = False,
        stochastic: bool = False,
    ):
        super(ConvGraph, self).__init__()

        self.grapher3d = Grapher(
            in_channels, act, bias, conv, dilation, dimension, drop_path, epsilon,
            kernel_size, n, norm, pick_y_method, pick_y_number, r,
            relative_pos, stochastic,
        )
        self.ffn3d = FFN(
            in_channels, act, dimension,
            drop_path,
            in_channels * 4, out_channels,
        )

    def forward(self, x):
        x = self.grapher3d(x)
        x = self.ffn3d(x)
        return x


def test_conv_graph_3d():
    import torch

    model = ConvGraph(16, 32, n=8).cuda()
    x = torch.randn(1, 16, 32, 32, 32).cuda()
    y = model(x)
    print(y.shape)


def test_conv_graph_2d():
    import torch

    model = ConvGraph(16, 32, n=8, dimension=2).cuda()
    x = torch.randn(1, 16, 32, 32).cuda()
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    test_conv_graph_3d()
    test_conv_graph_2d()

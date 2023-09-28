
from typing import Tuple, Union

import add_sys_path
import numpy as np
import torch
import torch.nn.functional as F
from gcn_lib.dygraphconv import DyGraphConv
from gcn_lib.pos_embed import (get_2d_relative_pos_embed,
                               get_3d_relative_pos_embed)
from timm.models.layers import DropPath
from torch import nn

# from gcn_lib import (DyGraphConv3d, get_3d_relative_pos_embed)


def is_perfect_cube(x):
    x = abs(x)
    return int(round(x ** (1. / 3))) ** 3 == x


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    fc1 is in_channels input and in_channels output
    n is the number of nodes in the graph and is equal to H*W*D
    """

    def _create_relative_pos(self):
        embed_dim = self.channels
        grid_size = int(round(self.n ** (1. / self.dimension)))
        if self.dimension == 3:
            pos_embed = get_3d_relative_pos_embed(embed_dim, grid_size)
        else:
            pos_embed = get_2d_relative_pos_embed(embed_dim, grid_size)
        pos_embed = np.float32(pos_embed)
        relative_pos_tensor = torch.from_numpy(pos_embed)
        relative_pos_tensor = relative_pos_tensor.unsqueeze(0).unsqueeze(1)
        if self.pick_y_method == 'avg_pool':
            if self.dimension == 3:

                relative_pos_tensor = F.interpolate(
                    relative_pos_tensor,
                    size=(self.n, self.n // (self.r * self.r * self.r)),
                    mode='bicubic', align_corners=False
                )
            else:
                relative_pos_tensor = F.interpolate(
                    relative_pos_tensor,
                    size=(self.n, self.n // (self.r * self.r)),
                    mode='bicubic', align_corners=False
                )
            self.relative_pos = nn.Parameter(
                -relative_pos_tensor.squeeze(1), requires_grad=False
            )
            return None
        elif self.pick_y_method == 'importance':
            self.relative_pos = nn.Parameter(
                -relative_pos_tensor.squeeze(1), requires_grad=False
            )
            return None
        raise NotImplementedError

    def _set_conv_bn(self):
        if self.dimension == 3:
            self.conv = nn.Conv3d
            self.bn = nn.BatchNorm3d
            self.pool = F.avg_pool3d
        elif self.dimension == 2:
            self.conv = nn.Conv2d
            self.bn = nn.BatchNorm2d
            self.pool = F.avg_pool2d
        else:
            raise ValueError(
                f'dimension must be 2 or 3, but got {self.dimension}'
            )

    def __init__(
        self,
        in_channels: int,
        act: str = 'relu',
        bias: bool = True,
        conv: str = 'edge',
        dilation: int = 1,
        dimension: int = 3,
        drop_path: float = 0.0,
        epsilon: float = 0.0,
        kernel_size: int = 9,
        n: int = 196,
        norm: Union[None, str] = None,
        pick_y_method: str = 'avg_pool',
        pick_y_number: int = 100,
        r: int = 4,
        relative_pos: bool = False,
        stochastic: bool = False,
    ):
        super(Grapher, self).__init__()

        if dimension == 3:
            assert is_perfect_cube(n), 'n must be a perfect cube'

        self.channels = in_channels
        self.dimension = dimension
        self.n = n
        self.pick_y_method = pick_y_method
        self.r = r

        self._set_conv_bn()

        self.fc1 = nn.Sequential(
            self.conv(in_channels, in_channels, 1, stride=1, padding=0),
            self.bn(in_channels),
        )
        self.graph_conv = DyGraphConv(
            in_channels,
            in_channels * 2,
            act,
            bias,
            False,
            conv,
            dilation,
            dimension,
            epsilon,
            kernel_size, norm, pick_y_method, pick_y_number, r, relative_pos,
            stochastic,
        )
        self.fc2 = nn.Sequential(
            self.conv(in_channels * 2, in_channels, 1, stride=1, padding=0),
            self.bn(in_channels),
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.relative_pos = None
        if relative_pos:
            self._create_relative_pos()

    def __check_input_shape__(self, x) -> Tuple[int, int, int]:
        if self.dimension == 3:
            assert x.dim() == 5, 'x must be 5D'
            _, _, H, W, D = x.shape
        elif self.dimension == 2:
            assert x.dim() == 4, 'x must be 4D'
            _, _, H, W = x.shape
            D = 1
        else:
            raise ValueError(
                f'dimension must be 2 or 3, but got {self.dimension}'
            )
        # assert H == W == D, 'H, W, D must be equal'
        return H, W, D

    def __interpolate__(self, relative_pos, size):
        relative_pos = relative_pos.unsqueeze(0)
        relative_pos = F.interpolate(
            relative_pos, size=size, mode="bicubic"
        ).squeeze(0)
        return relative_pos

    def _get_relative_pos(self, H: int, W: int, D: int):
        if self.relative_pos is None:
            return self.relative_pos

        N = H * W * D
        if N == self.n:
            return self.relative_pos
        else:
            if self.pick_y_method == 'avg_pool':
                N_reduced = N // (self.r ** self.dimension)
                size = (N, N_reduced)
            elif self.pick_y_method == 'importance':
                size = (N, N)
            else:
                raise NotImplementedError
            return self.__interpolate__(self.relative_pos, size)

    def forward(self, x):
        H, W, D = self.__check_input_shape__(x)

        _tmp = x

        x = self.fc1(x)
        relative_pos = self._get_relative_pos(H, W, D)
        x = self.graph_conv(x, relative_pos)

        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


def test_grapher3d():
    for pick_y_method in ['importance', 'avg_pool']:
        for relative_pos in [True, False]:
            x = torch.randn(2, 16, 14, 14, 14).cuda()
            grapher = Grapher(
                16, r=2, n=14 * 14 * 14, pick_y_method=pick_y_method,
                relative_pos=relative_pos
            ).cuda()
            y = grapher(x)
            print(y.shape)


def test_importance():
    print('test_importance')
    for relative_pos in [True, False]:
        x = torch.randn(2, 16, 14, 14, 14).cuda()
        grapher = Grapher(
            16, r=2, n=12 * 12 * 12,
            pick_y_number=1000, pick_y_method='importance',
            relative_pos=relative_pos
        ).cuda()
        y = grapher(x)
        print(y.shape)


def test_2d():
    print('test_2d')
    for pick_y_method in ['importance', 'avg_pool']:
        for relative_pos in [True, False]:
            x = torch.randn(2, 16, 12, 12).cuda()
            grapher = Grapher(
                16,
                dimension=2,
                r=2, n=12 * 12,
                pick_y_number=10,
                pick_y_method=pick_y_method,
                relative_pos=relative_pos
            ).cuda()
            y = grapher(x)
            print(y.shape)


if __name__ == '__main__':
    test_grapher3d()
    test_importance()
    test_2d()

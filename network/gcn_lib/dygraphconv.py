import copy
import gc
import unittest
from typing import Union

import add_sys_path
import numpy as np
import torch
import torch.nn.functional as F
from gcn_lib.torch_edge import DenseDilatedKnnGraph
from gcn_lib.torch_vertex import GraphConv2d
from torch import nn


def free_memory(tensor):
    if tensor.is_cuda:
        # Detach the tensor from the computation graph
        tensor = tensor.detach()

        # Move the tensor to the CPU
        tensor = tensor.cpu()

    # Delete the tensor
    del tensor

    # Manually trigger garbage collection if necessary (usually not required)
    gc.collect()

    # To empty the GPU cache after deleting the tensor, you can call:
    torch.cuda.empty_cache()


def print_gpu_memory_usage():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please make sure you have an NVIDIA GPU and the GPU version of PyTorch installed.")
        return

    device = torch.device('cuda')
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)

    print(f"Total GPU Memory: {total_memory / (1024 ** 2):.2f} MB")
    print(f"Allocated GPU Memory: {allocated_memory / (1024 ** 2):.2f} MB")
    print(f"Reserved GPU Memory: {reserved_memory / (1024 ** 2):.2f} MB")


class DyGraphConv(GraphConv2d):
    """ Dynamic graph convolution layer for a 3D input.

    forward function:
    1. compute the candidate neighbors `y`

    we have two methods to compute the neighbors `y` of the 3D input `x`.

    (1) 'avg_pool': average pooling the 3D input `x` to get `y` according to the
    reduce factor `r`.

    (2) 'importance': compute the `importance` of each point in the 3D input `x`
    by the `__compute_importance` function,
    pick the top `pick_y_number` points with the highest `importance` 
    as the candidate neighbors `y`. (not finished yet)

    2. after get the candidate neighbors `y`, we compute the edge index of the 
    graph by the `DenseDilatedKnnGraph` module according to `kernel_size`.

    3. then we perform the graph convolution on the flatten 3D input `x` and 
    the flatten candidate neighbors `y`.
    """

    def _check_in_channels(self, in_channels):
        # Because the edge conv is based on the 2D conv, and the group argument
        # of the 2D conv is hard coded to 4. So we require the in_channels to
        # be divisible by 4.
        if in_channels % 4 != 0:
            raise ValueError(
                f'in_channels must be divisible by 4, but got {in_channels}'
            )

    def _check_pick_y_number(self, pick_y_number, kernel_size, dilation):
        if pick_y_number < kernel_size * dilation:
            raise ValueError(
                f'pick_y_number must be greater than or equal to kernel_size * dilation, but got {pick_y_number} and {kernel_size * dilation}'
            )

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
        in_channels,
        out_channels,
        act: str = 'relu',
        bias: bool = True,
        compute_coordinate: bool = False,
        conv: str = 'edge',
        dilation: int = 1,
        dimension: int = 3,
        epsilon: float = 0.0,
        kernel_size: int = 9,
        norm: Union[None, str] = None,
        pick_y_method: str = 'avg_pool',
        pick_y_number: int = 100,
        r: int = 4,
        relative_pos: bool = False,
        stochastic: bool = False,
    ):
        self._check_in_channels(in_channels)
        self._check_pick_y_number(pick_y_number, kernel_size, dilation)

        super(DyGraphConv, self).__init__(
            in_channels, out_channels, conv, act, norm, bias
        )

        self.compute_coordinate = compute_coordinate
        self.dimension = dimension
        self.d = dilation
        self.k = kernel_size
        self.pick_y_method = pick_y_method
        self.pick_y_number = pick_y_number
        self.r = r
        self.relative_pos = relative_pos

        self._set_conv_bn()

        self.dilated_knn_graph = DenseDilatedKnnGraph(
            kernel_size, dilation, stochastic, epsilon
        )

        if pick_y_method == 'importance':
            self.importance_fc = nn.Sequential(
                self.conv(in_channels, in_channels, 1, stride=1, padding=0),
                self.bn(in_channels),
                nn.ReLU(inplace=True),
                self.conv(in_channels, 1, 1, stride=1, padding=0),
                self.bn(1),
                nn.Sigmoid(),
            )

    def _check_relative_pos_meet_requirement(self, relative_pos):
        if (relative_pos is None) and (self.relative_pos):
            raise ValueError('require relative position')
        if (relative_pos is not None) and (not self.relative_pos):
            raise ValueError('not require relative position')

    def _create_x_coords(self, x, verbose):
        if self.compute_coordinate and (self.dimension == 3):
            print('compute 3d x coordinate') if verbose else None

            # Create coordinate arrays for each dimension
            B, _, L, H, W = x.shape

            l_coord, h_coord, w_coord = np.meshgrid(
                np.arange(L), np.arange(H), np.arange(W), indexing='ij'
            )

            # Stack the coordinate arrays along the first dimension
            coords = np.stack((l_coord, h_coord, w_coord)) / max(L, H, W)
            x_coords = torch.from_numpy(coords).float().to(x.device)
            x_coords = torch.unsqueeze(x_coords, 0).repeat(B, 1, 1, 1, 1)

        elif self.compute_coordinate and (self.dimension == 2):
            print('compute 2d x coordinate') if verbose else None

            # Create coordinate arrays for each dimension
            B, _, H, W = x.shape

            h_coord, w_coord = np.meshgrid(
                np.arange(H), np.arange(W), indexing='ij'
            )

            # Stack the coordinate arrays along the first dimension
            coords = np.stack((h_coord, w_coord)) / max(H, W)
            x_coords = torch.from_numpy(coords).float().to(x.device)
            x_coords = torch.unsqueeze(x_coords, 0).repeat(B, 1, 1, 1)

        else:
            print('not compute x coordinate') if verbose else None
            x_coords = None

        return x_coords

    def _pool_y(self, x, x_coords):
        B, C = x.shape[0], x.shape[1]

        y = None
        y_coords = copy.deepcopy(x_coords)

        if self.r > 1:
            y = self.pool(x, kernel_size=self.r, stride=self.r)
            y = y.reshape(B, C, -1, 1).contiguous()

            if self.compute_coordinate:
                assert x_coords is not None
                if self.dimension == 3:
                    y_coords = y_coords[:, :, ::self.r, ::self.r, ::self.r]
                elif self.dimension == 2:
                    y_coords = y_coords[:, :, ::self.r, ::self.r]
                else:
                    raise ValueError(
                        f'dimension must be 2 or 3, but got {self.dimension}'
                    )

        return y, y_coords

    def _pick_y(self, x, x_coords):
        B, C = x.shape[0], x.shape[1]

        importance = self._importance.reshape(B, -1).contiguous()
        _, indices = torch.topk(importance, self.pick_y_number, dim=1)

        y_indices = indices.unsqueeze(1).repeat(1, C, 1)

        imp_x = x * self._importance
        imp_x = imp_x.reshape(B, C, -1).contiguous()
        y = torch.gather(imp_x, 2, y_indices)

        if self.compute_coordinate:
            y_coords_indices = indices.unsqueeze(1).repeat(
                1, self.dimension, 1
            )
            x_coords = x_coords.reshape(B, self.dimension, -1).contiguous()
            y_coords = torch.gather(x_coords, 2, y_coords_indices)
        else:
            y_coords = None

        return indices, y, y_coords

    def _select_relative_pos(self, relative_pos, indices):
        batch_size = indices.shape[0]
        list_relative_pos = []
        for i in range(batch_size):
            list_relative_pos.append(relative_pos[:, :, indices[i]])

        relative_pos = torch.cat(list_relative_pos, dim=0)
        return relative_pos

    def _compute_y(self, relative_pos, x, x_coords):
        self._importance = self.importance_fc(x)

        indices, y, y_coords = self._pick_y(x, x_coords)

        if relative_pos is not None:
            relative_pos = self._select_relative_pos(relative_pos, indices)

        return relative_pos, y, y_coords

    def _get_edge_coords(self, edge_index, x_coords, y_coords):
        _, B, num_x, num_y = edge_index.shape
        y_edge_index = edge_index[0]
        y_edge_index = y_edge_index.unsqueeze(1).repeat(
            1, self.dimension, 1, 1
        )
        y_edge_index = y_edge_index.reshape(B, self.dimension, -1)

        neighbor_coords = torch.gather(
            input=y_coords, dim=2, index=y_edge_index
        )
        neighbor_coords = neighbor_coords.reshape(
            B, self.dimension, num_x, num_y
        )

        edge_coords = torch.cat([x_coords, neighbor_coords], dim=3)

        return edge_coords.detach().cpu().numpy()

    def _check_x_shape(self, x):
        if self.dimension == 3:
            assert len(x.shape) == 5
        elif self.dimension == 2:
            assert len(x.shape) == 4
        else:
            raise NotImplementedError

    def forward(self, x, relative_pos=None, verbose=False):
        self._check_relative_pos_meet_requirement(relative_pos)
        self._check_x_shape(x)

        x_shape = x.shape
        B = x_shape[0]
        C = x_shape[1]

        x_coords = self._create_x_coords(x, verbose)

        if self.pick_y_method == 'avg_pool':
            print('pool y') if verbose else None
            y, y_coords = self._pool_y(x, x_coords)
        elif self.pick_y_method == 'importance':
            print('compute y') if verbose else None
            relative_pos, y, y_coords = self._compute_y(
                relative_pos, x, x_coords
            )
        else:
            raise NotImplementedError

        x = x.reshape(B, C, -1, 1).contiguous()
        y = y.reshape(B, C, -1, 1).contiguous() if y is not None else None
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv, self).forward(x, edge_index, y)

        if self.compute_coordinate:
            assert x_coords is not None
            assert y_coords is not None

            x_coords = x_coords.reshape(B, self.dimension, -1, 1).contiguous()
            y_coords = y_coords.reshape(B, self.dimension, -1).contiguous()

            self._edge_coords = self._get_edge_coords(
                edge_index, x_coords, y_coords
            )
            free_memory(x_coords)
            free_memory(y_coords)

        x_shape = list(x_shape)
        x_shape[1] = x.shape[1]
        return x.reshape(x_shape).contiguous()

    @property
    def importance(self):
        return self._importance

    @property
    def edge_coords(self):
        return self._edge_coords


class TestDyGraphConv3d(unittest.TestCase):

    def test_2d(self):
        print('>>> test 2d compute coordinate, importance, and relative pos')
        dy_graph_conv_3d = DyGraphConv(
            in_channels=4, out_channels=128, dimension=2, compute_coordinate=True,
            pick_y_method='importance', relative_pos=True
        )
        dy_graph_conv_3d.cuda()
        x = torch.rand(13, 4, 32, 32).cuda()
        relative_pos = torch.rand(1, 32 * 32, 32 * 32).cuda()
        output = dy_graph_conv_3d(x, relative_pos=relative_pos, verbose=True)
        self.assertEqual(output.shape, (13, 128, 32, 32))
        print_gpu_memory_usage()
        torch.cuda.empty_cache()

        print('>>> test 2d compute coordinate, avg pool, and relative pos')
        dy_graph_conv_3d = DyGraphConv(
            in_channels=4, out_channels=128, dimension=2, compute_coordinate=True,
            pick_y_method='avg_pool', relative_pos=True
        )
        dy_graph_conv_3d.cuda()
        x = torch.rand(13, 4, 32, 32).cuda()
        relative_pos = torch.rand(
            1, 32 * 32, int(32 * 32 / 16)
        ).cuda()
        output = dy_graph_conv_3d(x, relative_pos=relative_pos, verbose=True)
        self.assertEqual(output.shape, (13, 128, 32, 32))
        print_gpu_memory_usage()
        torch.cuda.empty_cache()

    def test_compute_coordinate_importance_rp(self):
        print('>>> test compute coordinate importance rp')
        dy_graph_conv_3d = DyGraphConv(
            in_channels=4, out_channels=128, compute_coordinate=True,
            pick_y_method='importance', relative_pos=True)
        dy_graph_conv_3d.cuda()
        x = torch.rand(13, 4, 32, 32, 32).cuda()
        relative_pos = torch.rand(1, 32 * 32 * 32, 32 * 32 * 32).cuda()
        output = dy_graph_conv_3d(x, relative_pos=relative_pos, verbose=True)
        self.assertEqual(output.shape, (13, 128, 32, 32, 32))

        print_gpu_memory_usage()
        torch.cuda.empty_cache()

    def test_compute_coordinate_importance_norp(self):
        # norp means no relative position
        print('>>> test compute coordinate importance norp')

        dy_graph_conv_3d = DyGraphConv(
            in_channels=4, out_channels=128, compute_coordinate=True,
            pick_y_method='importance')
        dy_graph_conv_3d.cuda()
        x = torch.rand(13, 4, 32, 32, 32).cuda()
        output = dy_graph_conv_3d(x)
        self.assertEqual(output.shape, (13, 128, 32, 32, 32))

        print_gpu_memory_usage()
        torch.cuda.empty_cache()

    def test_compute_coordinate_pool_rp(self):
        print('>>> test compute coordinate pool rp')
        dy_graph_conv_3d = DyGraphConv(
            in_channels=4, out_channels=128, compute_coordinate=True,
            pick_y_method='avg_pool', relative_pos=True)
        dy_graph_conv_3d.cuda()
        x = torch.rand(13, 4, 32, 32, 32).cuda()
        relative_pos = torch.rand(1, 32 * 32 * 32, 8 * 8 * 8).cuda()
        output = dy_graph_conv_3d(x, relative_pos=relative_pos, verbose=True)
        self.assertEqual(output.shape, (13, 128, 32, 32, 32))

        print_gpu_memory_usage()
        torch.cuda.empty_cache()

    def test_compute_coordinate_pool_norp(self):
        print('>>> test compute coordinate pool norp')
        dy_graph_conv_3d = DyGraphConv(
            in_channels=4, out_channels=128, compute_coordinate=True,
            pick_y_method='avg_pool')
        dy_graph_conv_3d.cuda()
        x = torch.rand(13, 4, 32, 32, 32).cuda()
        # relative_pos = torch.rand(13, 20, 3).cuda()
        output = dy_graph_conv_3d(x, verbose=True)
        self.assertEqual(output.shape, (13, 128, 32, 32, 32))

        print_gpu_memory_usage()
        torch.cuda.empty_cache()

    def test_importance_rp(self):
        print('>>> test importance rp')
        dy_graph_conv_3d = DyGraphConv(
            in_channels=4, out_channels=128,
            pick_y_method='importance', relative_pos=True)
        dy_graph_conv_3d.cuda()
        x = torch.rand(13, 4, 32, 32, 32).cuda()
        relative_pos = torch.rand(1, 32 * 32 * 32, 32 * 32 * 32).cuda()
        output = dy_graph_conv_3d(x, relative_pos=relative_pos, verbose=True)
        self.assertEqual(output.shape, (13, 128, 32, 32, 32))

        print_gpu_memory_usage()
        torch.cuda.empty_cache()

    def test_importance(self):
        # with no coordinate and no relative position
        print('>>> test importance')
        dy_graph_conv_3d = DyGraphConv(
            in_channels=4,
            out_channels=128,
            kernel_size=9,
            dilation=1,
            conv='edge',
            act='relu',
            norm=None,
            bias=True,
            stochastic=False,
            epsilon=0.0,
            r=1,
            pick_y_method='importance',
            pick_y_number=20
        )
        dy_graph_conv_3d.cuda()
        x = torch.rand(13, 4, 32, 32, 32).cuda()
        output = dy_graph_conv_3d(x)
        self.assertEqual(output.shape, (13, 128, 32, 32, 32))
        importance = dy_graph_conv_3d.importance
        self.assertEqual(importance.shape, (13, 1, 32, 32, 32))

        print_gpu_memory_usage()
        torch.cuda.empty_cache()

    def test_pool(self):
        dy_graph_conv_3d = DyGraphConv(
            in_channels=4,
            out_channels=128,
            kernel_size=9,
            dilation=1,
            conv='edge',
            act='relu',
            norm=None,
            bias=True,
            stochastic=False,
            epsilon=0.0,
            r=1,
            pick_y_method='avg_pool'
        )
        dy_graph_conv_3d.cuda()
        self.assertIsNotNone(dy_graph_conv_3d)

    def test_pool_rp(self):
        print('>>> test pool rp')
        dy_graph_conv_3d = DyGraphConv(
            in_channels=4, out_channels=128,
            pick_y_method='avg_pool', relative_pos=True)
        dy_graph_conv_3d.cuda()
        x = torch.rand(13, 4, 32, 32, 32).cuda()
        relative_pos = torch.rand(1, 32 * 32 * 32, 8 * 8 * 8).cuda()
        output = dy_graph_conv_3d(x, relative_pos=relative_pos, verbose=True)
        self.assertEqual(output.shape, (13, 128, 32, 32, 32))

        print_gpu_memory_usage()
        torch.cuda.empty_cache()

    def test_visualize_edge_coords(self):
        print('>>> test visualize edge coords')

        dy_graph_conv_3d = DyGraphConv(
            in_channels=4, out_channels=128, compute_coordinate=True,
            pick_y_method='importance')
        dy_graph_conv_3d.cuda()
        x = torch.rand(13, 4, 32, 32, 32).cuda()
        output = dy_graph_conv_3d(x)

        edge_coords = dy_graph_conv_3d.edge_coords

        from util.draw_connection import draw_connection
        edge_coords = edge_coords[0]
        edge_coords = np.transpose(edge_coords, (1, 2, 0))

        np.random.seed(0)
        for _ in range(10):
            i = np.random.randint(0, 32768)
            input_slice = x.detach().cpu().numpy()[0:1, :]
            centers = edge_coords[i, :] * 32
            centers = centers.astype(np.int32)
            draw_connection(input_slice, centers, f'test_{i}.png')


if __name__ == '__main__':
    import sys

    import add_sys_path
    print(sys.path)
    add_sys_path.void()

    unittest.main()

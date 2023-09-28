import add_sys_path
import torch.nn as nn
from gcn_lib.torch_nn import act_layer
from timm.models.layers import DropPath


class FFN(nn.Module):
    def _set_conv_bn(self):
        if self.dimension == 3:
            self.conv = nn.Conv3d
            self.bn = nn.BatchNorm3d
        elif self.dimension == 2:
            self.conv = nn.Conv2d
            self.bn = nn.BatchNorm2d
        else:
            raise ValueError(
                f'dimension must be 2 or 3, but got {self.dimension}'
            )

    def __init__(
        self,
        in_features,
        act='relu',
        dimension: int = 3,
        drop_path=0.0,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()

        self.dimension = dimension
        self._set_conv_bn()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            self.conv(in_features, hidden_features, 1, stride=1, padding=0),
            self.bn(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            self.conv(hidden_features, out_features, 1, stride=1, padding=0),
            self.bn(out_features),
        )
        self.drop_path = DropPath(
            drop_path
        ) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if shortcut.shape[1] != x.shape[1]:
            shortcut_shape = list(shortcut.shape)

            shortcut = shortcut.reshape(
                shortcut_shape[0], shortcut_shape[1], -1)

            shortcut = nn.functional.interpolate(
                shortcut.unsqueeze(0), size=(x.shape[1], shortcut.shape[-1]), mode='bicubic'
            ).squeeze(0)
            shortcut_shape[1] = x.shape[1]
            shortcut = shortcut.reshape(*shortcut_shape)

        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


def test_ffn3d():
    import torch
    ffn = FFN(
        64,
        act='relu',
        dimension=2,
        drop_path=0.1,
        hidden_features=64,
        out_features=3
    )
    x = torch.randn(1, 64, 14, 14)
    y = ffn(x)
    print(y.shape)


if __name__ == '__main__':
    test_ffn3d()

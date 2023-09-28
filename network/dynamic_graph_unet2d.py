import torch
import torch.nn as nn
import torch.nn.functional as F
from network.gcn_lib import ConvGraph
from network.superglue import (AttentionalGNN, KeypointEncoder, SuperGlue,
                               normalize_keypoints)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module,
                                                                                        nn.ConvTranspose2d) or isinstance(
                module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(
                module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def string_bool(input_string: str) -> bool:
    return bool(int(input_string))


def string_to_list(input_string, type, delimiter=','):
    if isinstance(input_string, list):
        return input_string
    return [type(x) for x in input_string.split(delimiter)]


class encoder(nn.Module):
    def __init__(
        self, in_channels, initial_filter_size, kernel_size, do_instancenorm,
        **kwargs
    ):
        super().__init__()

        use_graph_flags = kwargs.get('use_graph_flags', [False] * 8)
        use_graph_flags = string_to_list(use_graph_flags, type=string_bool)
        pick_y_methods = kwargs.get('pick_y_methods', ['avg_pool'] * 8)
        pick_y_methods = string_to_list(pick_y_methods, type=str)
        pick_y_numbers = kwargs.get('pick_y_numbers', [100] * 8)
        pick_y_numbers = string_to_list(pick_y_numbers, type=int)
        rs = kwargs.get('ratios', [2] * 8)
        rs = string_to_list(rs, type=int)

        self.contr_1_1 = self.contract(
            in_channels, initial_filter_size, kernel_size=kernel_size,
            instancenorm=do_instancenorm, use_dyanmic_graph=use_graph_flags[0],
            pick_y_method=pick_y_methods[0], pick_y_number=pick_y_numbers[0], r=rs[0]
        )
        self.contr_1_2 = self.contract(
            initial_filter_size, initial_filter_size, kernel_size=kernel_size,
            instancenorm=do_instancenorm, use_dyanmic_graph=use_graph_flags[1],
            pick_y_method=pick_y_methods[1], pick_y_number=pick_y_numbers[1], r=rs[1]
        )
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(
            initial_filter_size, initial_filter_size * 2, kernel_size=kernel_size,
            instancenorm=do_instancenorm, use_dyanmic_graph=use_graph_flags[2],
            pick_y_method=pick_y_methods[2], pick_y_number=pick_y_numbers[2], r=rs[2]
        )
        self.contr_2_2 = self.contract(
            initial_filter_size * 2, initial_filter_size * 2, kernel_size=kernel_size,
            instancenorm=do_instancenorm, use_dyanmic_graph=use_graph_flags[3],
            pick_y_method=pick_y_methods[3], pick_y_number=pick_y_numbers[3], r=rs[3]
        )

        self.contr_3_1 = self.contract(
            initial_filter_size * 2, initial_filter_size * 2 ** 2, kernel_size=kernel_size,
            instancenorm=do_instancenorm, use_dyanmic_graph=use_graph_flags[4],
            pick_y_method=pick_y_methods[4], pick_y_number=pick_y_numbers[4], r=rs[4]
        )
        self.contr_3_2 = self.contract(
            initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size=kernel_size,
            instancenorm=do_instancenorm, use_dyanmic_graph=use_graph_flags[5],
            pick_y_method=pick_y_methods[5], pick_y_number=pick_y_numbers[5], r=rs[5]
        )

        self.contr_4_1 = self.contract(
            initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 3, kernel_size=kernel_size,
            instancenorm=do_instancenorm, use_dyanmic_graph=use_graph_flags[6],
            pick_y_method=pick_y_methods[6], pick_y_number=pick_y_numbers[6], r=rs[6]
        )
        self.contr_4_2 = self.contract(
            initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size=kernel_size,
            instancenorm=do_instancenorm, use_dyanmic_graph=use_graph_flags[7],
            pick_y_method=pick_y_methods[7], pick_y_number=pick_y_numbers[7], r=rs[7]
        )
        # self.contr_4_3 = self.contract(
        #     initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size=kernel_size,
        #     instancenorm=do_instancenorm, use_dyanmic_graph=False,
        #     pick_y_method=pick_y_methods[7], pick_y_number=pick_y_numbers[7], r=rs[7]
        # )

        self.center = nn.Sequential(
            nn.Conv2d(
                initial_filter_size * 2 ** 3,
                initial_filter_size * 2 ** 4, 3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                initial_filter_size * 2 ** 4,
                initial_filter_size * 2 ** 4, 3, padding=1
            ),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def contract(
        in_channels, out_channels, instancenorm=True, kernel_size=3,
        pick_y_method='avg_pool', pick_y_number=100,
        use_dyanmic_graph=False, r=4,
    ):

        conv_graph = ConvGraph(
            in_channels, out_channels, dimension=2, pick_y_method=pick_y_method,
            pick_y_number=pick_y_number, r=r,
        ) if use_dyanmic_graph else None
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=1
        ) if not use_dyanmic_graph else None

        if instancenorm:
            if use_dyanmic_graph:
                layer = nn.Sequential(
                    conv_graph,
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(inplace=True)
                )
            else:
                layer = nn.Sequential(
                    conv2d,
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(inplace=True)
                )
        else:
            if use_dyanmic_graph:
                layer = nn.Sequential(
                    conv_graph,
                    nn.LeakyReLU(inplace=True)
                )
            else:
                layer = nn.Sequential(
                    conv2d,
                    nn.LeakyReLU(inplace=True)
                )
        return layer

    def forward(self, x):
        contr_1 = self.contr_1_2(self.contr_1_1(x))  # 512
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))  # 256
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))  # 128
        pool = self.pool(contr_3)

        # tmp_4 = self.contr_4_1(pool)
        # contr_4_2 = self.contr_4_2(tmp_4)  # 64
        # contr_4_3 = self.contr_4_3(tmp_4)
        # contr_4 = 0.5 * contr_4_2 + 0.5 * contr_4_3
        contr_4 = self.contr_4_2(self.contr_4_1(pool))  # 64
        pool = self.pool(contr_4)

        out = self.center(pool)
        return out, contr_4, contr_3, contr_2, contr_1


class decoder(nn.Module):
    def __init__(self, initial_filter_size, classes):
        super().__init__()
        # self.concat_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.upscale5 = nn.ConvTranspose2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3, kernel_size=2,
                                           stride=2)
        self.expand_4_1 = self.expand(
            initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
        self.expand_4_2 = self.expand(
            initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2, kernel_size=2,
                                           stride=2)

        self.expand_3_1 = self.expand(
            initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_2 = self.expand(
            initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)
        self.upscale3 = nn.ConvTranspose2d(
            initial_filter_size * 2 ** 2, initial_filter_size * 2, 2, stride=2)

        self.expand_2_1 = self.expand(
            initial_filter_size * 2 ** 2, initial_filter_size * 2)
        self.expand_2_2 = self.expand(
            initial_filter_size * 2, initial_filter_size * 2)
        self.upscale2 = nn.ConvTranspose2d(
            initial_filter_size * 2, initial_filter_size, 2, stride=2)

        self.expand_1_1 = self.expand(
            initial_filter_size * 2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        self.head = nn.Sequential(
            nn.Conv2d(initial_filter_size, classes, kernel_size=1,
                      stride=1, bias=False))

    def forward(self, x, contr_4, contr_3, contr_2, contr_1):

        concat_weight = 1
        upscale = self.upscale5(x)
        crop = self.center_crop(contr_4, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_2_2(self.expand_2_1(concat))
        upscale = self.upscale2(expand)

        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_1_2(self.expand_1_1(concat))

        out = self.head(expand)
        return out

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        return layer


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=4, do_instancenorm=True,
                 **kwargs):
        super().__init__()

        self.encoder = encoder(
            in_channels, initial_filter_size, kernel_size, do_instancenorm, **kwargs)
        self.decoder = decoder(initial_filter_size, classes)

        self.apply(InitWeights_He(1e-2))

    def forward(self, x):

        x_1, contr_4, contr_3, contr_2, contr_1 = self.encoder(x)
        out = self.decoder(x_1, contr_4, contr_3, contr_2, contr_1)

        return out


class UNet2D_classification(nn.Module):
    def __init__(
        self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=3, do_instancenorm=True,
        **kwargs
    ):
        super().__init__()

        self.encoder = encoder(
            in_channels, initial_filter_size, kernel_size, do_instancenorm,
            **kwargs
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(initial_filter_size * 2 ** 4,
                      initial_filter_size * 2 ** 4),
            nn.ReLU(inplace=True),
            nn.Linear(initial_filter_size * 2 ** 4, classes)
        )

        self.apply(InitWeights_He(1e-2))

    def forward(self, x):

        x_1, _, _, _, _ = self.encoder(x)
        out = self.head(x_1)

        return out


class GraphUNet2D_classification(nn.Module):
    def __init__(
        self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=3, do_instancenorm=True,
        **kwargs
    ):
        super().__init__()

        self.encoder = encoder(
            in_channels, initial_filter_size, kernel_size, do_instancenorm,
            **kwargs
        )

        self.superglue = SuperGlue(
            {
                'sinkhorn_iterations': 100,
                'match_threshold': 0.2,
                'num_layers': 9,
                'use_layernorm': False,
                'bin_value': 1.0,
                'pos_loss_weight': 0.45,
                'neg_loss_weight': 1.0
            }
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(initial_filter_size * 2 ** 4,
                      initial_filter_size * 2 ** 4),
            nn.ReLU(inplace=True),
            nn.Linear(initial_filter_size * 2 ** 4, classes)
        )

        self.graph_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, initial_filter_size * 2 ** 4)
        )

        self.feature_mlp = nn.Conv1d(32, 256, kernel_size=1, bias=True)

        self.apply(InitWeights_He(1e-2))

    def _gather_index(self, feature, position):
        index = position[..., 0] * feature.shape[-1] + position[..., 1]
        index = index.long()
        index = index.unsqueeze(-2).repeat(1, feature.shape[1], 1)
        output = torch.gather(
            feature.view(feature.shape[0], feature.shape[1], -1), 2, index
        )
        return output

    def _gather_features(self, contr_1, keypoints):
        keypoints = keypoints.long()
        output_1 = self._gather_index(contr_1, keypoints)
        output = output_1
        return output

    def forward(self, x, mode='train'):

        images0 = x['images_1']
        images1 = x['images_2']
        keypoints0 = x['keypoints_1']
        keypoints1 = x['keypoints_2']
        all_matches = []
        for each_gpu_batch_i, batch_i in enumerate(x['batch_index']):
            batch_i = batch_i.item()
            match_i = x['matches_list'][batch_i]
            match_i[:, 0] = each_gpu_batch_i
            all_matches.append(match_i.long())
        match_indexes = torch.cat(all_matches, dim=0)
        # match_indexes = x['matches']
        gt_vector = x['gt_vec']

        # x_1, _, _, _, _ = self.encoder(x)

        x_1_0, contr_4, contr_3, contr_2, contr_1 = self.encoder(images0)
        out_0 = self.head(x_1_0)

        descriptors0 = self._gather_features(contr_1, keypoints0)
        descriptors0 = self.feature_mlp(descriptors0)

        x_1_1, contr_4, contr_3, contr_2, contr_1 = self.encoder(images1)
        out_1 = self.head(x_1_1)

        descriptors1 = self._gather_features(contr_1, keypoints1)
        descriptors1 = self.feature_mlp(descriptors1)

        superglue_input = {
            'keypoints0': keypoints0, 'keypoints1': keypoints1,
            'descriptors0': descriptors0, 'descriptors1': descriptors1,
            'image0': images0, 'image1': images1,
            'scores0': None, 'scores1': None,
            'matches': match_indexes,
            'gt_vec': gt_vector
        }
        if mode is 'train':
            superglue_total_loss, pos_loss, neg_loss, desc0, desc1 = self.superglue(
                superglue_input, **{'mode': mode}
            )

            # out = self.head(x_1)
            desc0 = torch.mean(desc0, dim=2)
            desc1 = torch.mean(desc1, dim=2)
            desc0 = self.graph_head(desc0)
            desc1 = self.graph_head(desc1)

            return superglue_total_loss, out_0, out_1, desc0, desc1
        else:
            batch_return_dict = self.superglue(
                superglue_input, **{'mode': mode}
            )

            return batch_return_dict


class GraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()

        self.feature_mlp = nn.Conv1d(
            in_channels, 256, kernel_size=1, bias=True
        )

        self.diffusion_cnn = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

        self.superglue = SuperGlue(
            {
                'sinkhorn_iterations': 100,
                'match_threshold': 0.2,
                'GNN_layers': ['self', 'cross'] * num_layers,
                'use_layernorm': False,
                'bin_value': 1.0,
                'pos_loss_weight': 0.45,
                'neg_loss_weight': 1.0
            }
        )

    def _gather_index(self, feature, position):
        position = position.long()
        index = position[..., 0] * feature.shape[-1] + position[..., 1]
        index = index.long()
        index = index.unsqueeze(-2).repeat(1, feature.shape[1], 1)
        output = torch.gather(feature.view(feature.shape[0],
                                           feature.shape[1], -1), 2, index)
        return output

    def assign_graph_features(self, contr_1, keypoints, graph_feature):
        zero_tensor = torch.zeros(
            contr_1.shape[0], graph_feature.shape[1],
            contr_1.shape[2], contr_1.shape[3],
            device=contr_1.device
        )
        zero_tensor_view = zero_tensor.view(
            contr_1.shape[0], graph_feature.shape[1], -1
        )
        keypoint_indices = keypoints[..., 0] * \
            contr_1.shape[3] + keypoints[..., 1]
        ki = keypoint_indices.unsqueeze(1).expand(
            -1, graph_feature.shape[1], -1
        )
        ki = ki.long()
        # Scatter graph_feature values into zero_tensor using the keypoint_indices
        graph_image_feature = zero_tensor_view.scatter_(
            2, ki, graph_feature
        )
        graph_image_feature = graph_image_feature.view(
            contr_1.shape[0], graph_feature.shape[1],
            contr_1.shape[2], contr_1.shape[3]
        )

        return graph_image_feature

    def forward(self, feature_2d, keypoints):

        graph_feature = self._gather_index(feature_2d, keypoints)
        graph_feature = self.feature_mlp(graph_feature)

        superglue_input = {
            'keypoints0': keypoints,
            'descriptors0': graph_feature,
            'image0': feature_2d,
            'scores0': None,
        }
        desc0 = self.superglue(
            superglue_input, **{'mode': 'sup'}
        )

        # merge graph feature and image feature
        graph_image_feature = self.assign_graph_features(
            feature_2d, keypoints, desc0
        )
        graph_image_feature = self.diffusion_cnn(graph_image_feature)

        return graph_image_feature, desc0


class GraphUNetEncoder(nn.Module):
    @staticmethod
    def contract(in_channels, out_channels, instancenorm=True, kernel_size=3,):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
        else:
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding=1
                ),
                nn.LeakyReLU(inplace=True)
            )
        return layer

    def __init__(self, in_channels, initial_filter_size, kernel_size, do_instancenorm,
                 **kwargs):
        super().__init__()

        num_layers = kwargs.get('num_layers', 8)

        # in_channels = 1
        # initial_filter_size = 32
        # kernel_size = 3
        # do_instancenorm = True

        self.contr_1_1 = self.contract(
            in_channels, initial_filter_size, kernel_size=kernel_size,
            instancenorm=do_instancenorm
        )
        self.contr_1_2 = self.contract(
            initial_filter_size, initial_filter_size, kernel_size=kernel_size,
            instancenorm=do_instancenorm,
        )

        self.graph_1 = GraphLayer(
            initial_filter_size, initial_filter_size, num_layers
        )

        self.contr_2_1 = self.contract(
            initial_filter_size * 2, initial_filter_size * 2, kernel_size=kernel_size,
            instancenorm=do_instancenorm,
        )
        self.contr_2_2 = self.contract(
            initial_filter_size * 2, initial_filter_size * 2, kernel_size=kernel_size,
            instancenorm=do_instancenorm,
        )
        self.graph_2 = GraphLayer(
            initial_filter_size * 2, initial_filter_size * 2, num_layers
        )

        self.contr_3_1 = self.contract(
            initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size=kernel_size,
            instancenorm=do_instancenorm,
        )
        self.contr_3_2 = self.contract(
            initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size=kernel_size,
            instancenorm=do_instancenorm,
        )
        self.graph_3 = GraphLayer(
            initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, num_layers
        )

        self.contr_4_1 = self.contract(
            initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size=kernel_size,
            instancenorm=do_instancenorm,
        )
        self.contr_4_2 = self.contract(
            initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size=kernel_size,
            instancenorm=do_instancenorm,
        )
        self.graph_4 = GraphLayer(
            initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, num_layers
        )

        self.center = nn.Sequential(
            nn.Conv2d(
                initial_filter_size * 2 ** 4,
                initial_filter_size * 2 ** 5, 3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                initial_filter_size * 2 ** 5,
                initial_filter_size * 2 ** 5, 3, padding=1
            ),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, image, keypoints):
        # block 1
        contr_1 = self.contr_1_2(self.contr_1_1(image))  # 512
        graph_1, sparse_graph_1 = self.graph_1(contr_1, keypoints)
        contr_1 = torch.cat([contr_1, graph_1], dim=1)
        pool = self.pool(contr_1)

        # block 2
        contr_2 = self.contr_2_2(self.contr_2_1(pool))  # 256
        graph_2, sparse_graph_2 = self.graph_2(
            contr_2, torch.div(keypoints, 2, rounding_mode='trunc')
        )
        contr_2 = torch.cat([contr_2, graph_2], dim=1)
        pool = self.pool(contr_2)

        # block 3
        contr_3 = self.contr_3_2(self.contr_3_1(pool))  # 128
        graph_3, sparse_graph_3 = self.graph_3(
            contr_3, torch.div(keypoints, 4, rounding_mode='trunc')
        )
        contr_3 = torch.cat([contr_3, graph_3], dim=1)
        pool = self.pool(contr_3)

        # block 4
        contr_4 = self.contr_4_2(self.contr_4_1(pool))  # 64
        graph_4, sparse_graph_4 = self.graph_4(
            contr_4, torch.div(keypoints, 8, rounding_mode='trunc')
        )
        contr_4 = torch.cat([contr_4, graph_4], dim=1)
        pool = self.pool(contr_4)

        out = self.center(pool)

        sparse_graph_feature = torch.cat(
            [
                sparse_graph_1, sparse_graph_2, sparse_graph_3, sparse_graph_4
            ],
            dim=1
        )

        return out, contr_4, contr_3, contr_2, contr_1, sparse_graph_feature


class GraphUNet2DClassification(nn.Module):
    ''' this class is created for better utilize the graph information
    '''

    def __init__(
        self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=3, do_instancenorm=True,
        **kwargs
    ):
        super().__init__()

        self.encoder = GraphUNetEncoder(
            in_channels, initial_filter_size, kernel_size, do_instancenorm,
            **kwargs
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(initial_filter_size * 2 ** 5,
                      initial_filter_size * 2 ** 4),
            nn.ReLU(inplace=True),
            nn.Linear(initial_filter_size * 2 ** 4, classes)
        )

        self.graph_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, classes)
        )

        self.superglue = SuperGlue(
            {
                # 'descriptor_dim': 64,
                'sinkhorn_iterations': 100,
                'match_threshold': 0.2,
                'GNN_layers': ['self', 'cross'] * 2,
                'use_layernorm': False,
                'bin_value': 1.0,
                'pos_loss_weight': 0.45,
                'neg_loss_weight': 1.0
            }
        )

        self.graph_feature_mlp = nn.Conv1d(
            256 * 4, 256, kernel_size=1, bias=True
        )

        self.graph_local_head = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, classes, kernel_size=1)
        )

        self.apply(InitWeights_He(1e-2))

    def forward(self, input_dict, weight_corr=1.0, weight_local=1.0):
        # gather the matches and gt_vector
        all_matches = []
        for each_gpu_batch_i, batch_i in enumerate(input_dict['batch_index']):
            batch_i = batch_i.item()
            match_i = input_dict['matches_list'][batch_i]
            match_i[:, 0] = each_gpu_batch_i
            all_matches.append(match_i.long())
        match_indexes = torch.cat(all_matches, dim=0)
        gt_vector = input_dict['gt_vec']

        # get image feature and graph feature
        images_1 = input_dict['images_1']
        keypoints_1 = input_dict['keypoints_1']
        x_1, _, _, _, _, graph_1 = self.encoder(images_1, keypoints_1)
        graph_1 = self.graph_feature_mlp(graph_1)

        # get image feature and graph feature
        images_2 = input_dict['images_2']
        keypoints_2 = input_dict['keypoints_2']
        x_2, _, _, _, _, graph_2 = self.encoder(images_2, keypoints_2)
        graph_2 = self.graph_feature_mlp(graph_2)

        # superglue
        if weight_corr > 0.:
            superglue_input = {
                'keypoints0': keypoints_1, 'keypoints1': keypoints_2,
                'descriptors0': graph_1, 'descriptors1': graph_2,
                'image0': images_1, 'image1': images_2,
                'scores0': None, 'scores1': None,
                'matches': match_indexes,
                'gt_vec': gt_vector
            }
            superglue_total_loss, _, _, _, _ = self.superglue(
                superglue_input, **{'mode': 'train'}
            )
        else:
            superglue_total_loss = torch.tensor(0.0, device=images_1.device)

        # prepare for graph constrastive loss
        desc0 = torch.mean(graph_1, dim=2)
        desc1 = torch.mean(graph_2, dim=2)
        desc0 = self.graph_head(desc0)
        desc1 = self.graph_head(desc1)

        # prepare of cnn contrastive loss
        out_1 = self.head(x_1)
        out_2 = self.head(x_2)

        # graph local contrastive loss
        if weight_local > 0.:
            local_graph_feature_1 = self.graph_local_head(graph_1)
            local_graph_feature_2 = self.graph_local_head(graph_2)
            mask = torch.zeros(
                local_graph_feature_1.shape[0],
                local_graph_feature_1.shape[2],
                local_graph_feature_2.shape[2],
                device=local_graph_feature_1.device
            )
            mask[match_indexes[:, 0], match_indexes[:, 1],
                 match_indexes[:, 2]] = 1.0
        else:
            local_graph_feature_1 = None
            local_graph_feature_2 = None
            mask = None

        # if weight_corr > 0. and weight_local <= 0.:
        #     return superglue_total_loss, out_1, out_2, desc0, desc1, mask
        return superglue_total_loss, out_1, out_2, desc0, desc1, mask, local_graph_feature_1, local_graph_feature_2


class GraphUnetV4(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=4, do_instancenorm=True, **kwargs) -> None:
        super().__init__()

        self.encoder = encoder(
            in_channels, initial_filter_size, kernel_size, do_instancenorm, **kwargs
        )
        self.decoder = decoder(initial_filter_size, classes)

        self.superglue = SuperGlue(
            {
                'sinkhorn_iterations': 100,
                'match_threshold': 0.2,
                'num_layers': 9,
                'use_layernorm': False,
                'bin_value': 1.0,
                'pos_loss_weight': 0.45,
                'neg_loss_weight': 1.0
            }
        )

        self.feature_mlp = nn.Conv1d(992, 256, kernel_size=1, bias=True)
        self.inverse_mlp = nn.Conv1d(256, 32, kernel_size=1, bias=True)

        self.diffusion_cnn = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
        )

        self.head = nn.Conv1d(256, classes, kernel_size=1, bias=True)

        self.apply(InitWeights_He(1e-2))

        self.graph_counter = 100000
        self.graph_steps = 100000

    def _gather_index(self, feature, position):
        index = position[..., 0] * feature.shape[-1] + position[..., 1]
        index = index.long()
        index = index.unsqueeze(-2).repeat(1, feature.shape[1], 1)
        output = torch.gather(feature.view(feature.shape[0],
                                           feature.shape[1], -1), 2, index)
        return output

    def _gather_features(self, contr_1, contr_2, contr_3, contr_4, x_1,
                         keypoints):
        keypoints = keypoints.long()
        # output_1 = contr_1[:, :, keypoints[:, :, 0], keypoints[:, :, 1]]
        # contr_1_swapped = contr_1.permute(0, 2, 3, 1)
        # output_1 = contr_1_swapped[keypoints]
        output_1 = self._gather_index(contr_1, keypoints)
        output_2 = self._gather_index(contr_2,
                                      torch.div(keypoints, 2, rounding_mode='trunc'))
        output_3 = self._gather_index(contr_3,
                                      torch.div(keypoints, 4, rounding_mode='trunc'))
        output_4 = self._gather_index(contr_4,
                                      torch.div(keypoints, 8, rounding_mode='trunc'))
        output_5 = self._gather_index(x_1,
                                      torch.div(keypoints, 16, rounding_mode='trunc'))

        output = torch.cat(
            [output_1, output_2, output_3, output_4, output_5], dim=1)
        # output = output_1

        return output

    def assign_graph_features(self, contr_1, keypoints, graph_feature):
        zero_tensor = torch.zeros(
            contr_1.shape[0], graph_feature.shape[1],
            contr_1.shape[2], contr_1.shape[3],
            device=contr_1.device
        )
        zero_tensor_view = zero_tensor.view(
            contr_1.shape[0], graph_feature.shape[1], -1
        )
        keypoint_indices = keypoints[..., 0] * \
            contr_1.shape[3] + keypoints[..., 1]
        ki = keypoint_indices.unsqueeze(1).expand(
            -1, graph_feature.shape[1], -1
        )
        ki = ki.long()
        # Scatter graph_feature values into zero_tensor using the keypoint_indices
        graph_image_feature = zero_tensor_view.scatter_(
            2, ki, graph_feature
        )
        graph_image_feature = graph_image_feature.view(
            contr_1.shape[0], graph_feature.shape[1],
            contr_1.shape[2], contr_1.shape[3]
        )

        return graph_image_feature

    def forward(self, x, keypoints, x_11, keypoints_1, mode='val'):

        x_1, contr_4, contr_3, contr_2, contr_1 = self.encoder(x)
        x_1_1, contr_4_1, contr_3_1, contr_2_1, contr_1_1 = self.encoder(x_11)

        graph_feature = self._gather_features(
            contr_1, contr_2, contr_3, contr_4, x_1,
            keypoints
        )
        graph_feature = self.feature_mlp(graph_feature)

        # graph_feature_1 = self._gather_features(contr_1_1, keypoints_1)
        # graph_feature_1 = self.feature_mlp(graph_feature_1)

        superglue_input = {
            'keypoints0': keypoints,
            'descriptors0': graph_feature,
            'image0': x,
            'scores0': None,
        }
        desc0 = self.superglue(
            superglue_input, **{'mode': 'sup'}
        )

        # merge graph feature and image feature
        # graph_feature = self.inverse_mlp(desc0)
        graph_image_feature = self.assign_graph_features(
            contr_1, keypoints, desc0
        )
        graph_image_feature = self.diffusion_cnn(graph_image_feature)

        if mode == 'train':
            self.graph_counter += 1
        graph_weight = self.graph_counter / self.graph_steps
        if graph_weight > 1:
            graph_weight = 1
        print(f'graph_weight: {graph_weight}')
        contr_1 = contr_1 + graph_weight * graph_image_feature

        graph_out = self.head(desc0)

        out = self.decoder(x_1, contr_4, contr_3, contr_2, contr_1)

        if keypoints is None:
            return out

        return out, graph_out


class GraphUnetV5(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=32, kernel_size=3, classes=4, do_instancenorm=True, **kwargs) -> None:
        super().__init__()

        self.encoder = GraphUNetEncoder(
            in_channels, initial_filter_size, kernel_size, do_instancenorm,
            **kwargs
        )
        self.decoder = decoder(initial_filter_size * 2, classes)

        self.apply(InitWeights_He(1e-2))

    def forward(self, x, keypoints):

        x_1, contr_4, contr_3, contr_2, contr_1, _ = self.encoder(x, keypoints)
        out = self.decoder(x_1, contr_4, contr_3, contr_2, contr_1)

        return out, None


if __name__ == '__main__':
    model = UNet2D(in_channels=1, initial_filter_size=32,
                   kernel_size=3, classes=3, do_instancenorm=True)
    input = torch.randn(5, 1, 160, 160)
    out = model(input)
    print(f'out shape:{out.shape}')

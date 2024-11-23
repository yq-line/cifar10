# import torch
# import torch.nn as nn
# import math

# from collections import OrderedDict
# from functools import partial
# from typing import Callable, Optional

# import torch.nn as nn
# import torch
# from torch import Tensor


# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """
#     Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

#     This function is taken from the rwightman.
#     It can be seen here:
#     https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output


# class DropPath(nn.Module):
#     """
#     Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)


# class ConvBNAct(nn.Module):
#     def __init__(self,
#                  in_planes: int,
#                  out_planes: int,
#                  kernel_size: int = 3,
#                  stride: int = 1,
#                  groups: int = 1,
#                  norm_layer: Optional[Callable[..., nn.Module]] = None,
#                  activation_layer: Optional[Callable[..., nn.Module]] = None):
#         super(ConvBNAct, self).__init__()

#         padding = (kernel_size - 1) // 2
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if activation_layer is None:
#             activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

#         self.conv = nn.Conv2d(in_channels=in_planes,
#                               out_channels=out_planes,
#                               kernel_size=kernel_size,
#                               stride=stride,
#                               padding=padding,
#                               groups=groups,
#                               bias=False)

#         self.bn = norm_layer(out_planes)
#         self.act = activation_layer()

#     def forward(self, x):
#         result = self.conv(x)
#         result = self.bn(result)
#         result = self.act(result)

#         return result


# class SqueezeExcite(nn.Module):
#     def __init__(self,
#                  input_c: int,   # block input channel
#                  expand_c: int,  # block expand channel
#                  se_ratio: float = 0.25):
#         super(SqueezeExcite, self).__init__()
#         squeeze_c = int(input_c * se_ratio)
#         self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
#         self.act1 = nn.SiLU()  # alias Swish
#         self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
#         self.act2 = nn.Sigmoid()

#     def forward(self, x: Tensor) -> Tensor:
#         scale = x.mean((2, 3), keepdim=True)
#         scale = self.conv_reduce(scale)
#         scale = self.act1(scale)
#         scale = self.conv_expand(scale)
#         scale = self.act2(scale)
#         return scale * x


# class MBConv(nn.Module):
#     def __init__(self,
#                  kernel_size: int,
#                  input_c: int,
#                  out_c: int,
#                  expand_ratio: int,
#                  stride: int,
#                  se_ratio: float,
#                  drop_rate: float,
#                  norm_layer: Callable[..., nn.Module]):
#         super(MBConv, self).__init__()

#         if stride not in [1, 2]:
#             raise ValueError("illegal stride value.")

#         self.has_shortcut = (stride == 1 and input_c == out_c)

#         activation_layer = nn.SiLU  # alias Swish
#         expanded_c = input_c * expand_ratio

#         # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
#         assert expand_ratio != 1
#         # Point-wise expansion
#         self.expand_conv = ConvBNAct(input_c,
#                                      expanded_c,
#                                      kernel_size=1,
#                                      norm_layer=norm_layer,
#                                      activation_layer=activation_layer)

#         # Depth-wise convolution
#         self.dwconv = ConvBNAct(expanded_c,
#                                 expanded_c,
#                                 kernel_size=kernel_size,
#                                 stride=stride,
#                                 groups=expanded_c,
#                                 norm_layer=norm_layer,
#                                 activation_layer=activation_layer)

#         self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

#         # Point-wise linear projection
#         self.project_conv = ConvBNAct(expanded_c,
#                                       out_planes=out_c,
#                                       kernel_size=1,
#                                       norm_layer=norm_layer,
#                                       activation_layer=nn.Identity)  # 注意这里没有激活函数，所有传入Identity

#         self.out_channels = out_c

#         # 只有在使用shortcut连接时才使用dropout层
#         self.drop_rate = drop_rate
#         if self.has_shortcut and drop_rate > 0:
#             self.dropout = DropPath(drop_rate)

#     def forward(self, x: Tensor) -> Tensor:
#         result = self.expand_conv(x)
#         result = self.dwconv(result)
#         result = self.se(result)
#         result = self.project_conv(result)

#         if self.has_shortcut:
#             if self.drop_rate > 0:
#                 result = self.dropout(result)
#             result += x

#         return result


# class FusedMBConv(nn.Module):
#     def __init__(self,
#                  kernel_size: int,
#                  input_c: int,
#                  out_c: int,
#                  expand_ratio: int,
#                  stride: int,
#                  se_ratio: float,
#                  drop_rate: float,
#                  norm_layer: Callable[..., nn.Module]):
#         super(FusedMBConv, self).__init__()

#         assert stride in [1, 2]
#         assert se_ratio == 0

#         self.has_shortcut = stride == 1 and input_c == out_c
#         self.drop_rate = drop_rate

#         self.has_expansion = expand_ratio != 1

#         activation_layer = nn.SiLU  # alias Swish
#         expanded_c = input_c * expand_ratio

#         # 只有当expand ratio不等于1时才有expand conv
#         if self.has_expansion:
#             # Expansion convolution
#             self.expand_conv = ConvBNAct(input_c,
#                                          expanded_c,
#                                          kernel_size=kernel_size,
#                                          stride=stride,
#                                          norm_layer=norm_layer,
#                                          activation_layer=activation_layer)

#             self.project_conv = ConvBNAct(expanded_c,
#                                           out_c,
#                                           kernel_size=1,
#                                           norm_layer=norm_layer,
#                                           activation_layer=nn.Identity)  # 注意没有激活函数
#         else:
#             # 当只有project_conv时的情况
#             self.project_conv = ConvBNAct(input_c,
#                                           out_c,
#                                           kernel_size=kernel_size,
#                                           stride=stride,
#                                           norm_layer=norm_layer,
#                                           activation_layer=activation_layer)  # 注意有激活函数

#         self.out_channels = out_c

#         # 只有在使用shortcut连接时才使用dropout层
#         self.drop_rate = drop_rate
#         if self.has_shortcut and drop_rate > 0:
#             self.dropout = DropPath(drop_rate)

#     def forward(self, x: Tensor) -> Tensor:
#         if self.has_expansion:
#             result = self.expand_conv(x)
#             result = self.project_conv(result)
#         else:
#             result = self.project_conv(x)

#         if self.has_shortcut:
#             if self.drop_rate > 0:
#                 result = self.dropout(result)

#             result += x

#         return result


# class EfficientNetV2(nn.Module):
#     def __init__(self,
#                  model_cnf: list,
#                  num_classes: int = 10,
#                  num_features: int = 1280,
#                  dropout_rate: float = 0.2,
#                  drop_connect_rate: float = 0.2):
#         super(EfficientNetV2, self).__init__()

#         for cnf in model_cnf:
#             assert len(cnf) == 8

#         norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

#         stem_filter_num = model_cnf[0][4]

#         self.stem = ConvBNAct(3,
#                               stem_filter_num,
#                               kernel_size=3,
#                               stride=2,
#                               norm_layer=norm_layer)  # 激活函数默认是SiLU

#         total_blocks = sum([i[0] for i in model_cnf])
#         block_id = 0
#         blocks = []
#         for cnf in model_cnf:
#             repeats = cnf[0]
#             op = FusedMBConv if cnf[-2] == 0 else MBConv
#             for i in range(repeats):
#                 blocks.append(op(kernel_size=cnf[1],
#                                  input_c=cnf[4] if i == 0 else cnf[5],
#                                  out_c=cnf[5],
#                                  expand_ratio=cnf[3],
#                                  stride=cnf[2] if i == 0 else 1,
#                                  se_ratio=cnf[-1],
#                                  drop_rate=drop_connect_rate * block_id / total_blocks,
#                                  norm_layer=norm_layer))
#                 block_id += 1
#         self.blocks = nn.Sequential(*blocks)

#         head_input_c = model_cnf[-1][-3]
#         head = OrderedDict()

#         head.update({"project_conv": ConvBNAct(head_input_c,
#                                                num_features,
#                                                kernel_size=1,
#                                                norm_layer=norm_layer)})  # 激活函数默认是SiLU

#         head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
#         head.update({"flatten": nn.Flatten()})

#         if dropout_rate > 0:
#             head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
#         head.update({"classifier": nn.Linear(num_features, num_classes)})

#         self.head = nn.Sequential(head)

#         # initial weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out")
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.stem(x)
#         x = self.blocks(x)
#         x = self.head(x)

#         return x


# def efficientnetv2_s(num_classes: int = 10):
#     """
#     EfficientNetV2
#     https://arxiv.org/abs/2104.00298
#     """
#     # train_size: 300, eval_size: 384

#     # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
#     model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
#                     [4, 3, 2, 4, 24, 48, 0, 0],
#                     [4, 3, 2, 4, 48, 64, 0, 0],
#                     [6, 3, 2, 4, 64, 128, 1, 0.25],
#                     [9, 3, 1, 6, 128, 160, 1, 0.25],
#                     [15, 3, 2, 6, 160, 256, 1, 0.25]]

#     model = EfficientNetV2(model_cnf=model_config,
#                            num_classes=num_classes,
#                            dropout_rate=0.2)
#     return model


# def efficientnetv2_m(num_classes: int = 10):
#     """
#     EfficientNetV2
#     https://arxiv.org/abs/2104.00298
#     """
#     # train_size: 384, eval_size: 480

#     # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
#     model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
#                     [5, 3, 2, 4, 24, 48, 0, 0],
#                     [5, 3, 2, 4, 48, 80, 0, 0],
#                     [7, 3, 2, 4, 80, 160, 1, 0.25],
#                     [14, 3, 1, 6, 160, 176, 1, 0.25],
#                     [18, 3, 2, 6, 176, 304, 1, 0.25],
#                     [5, 3, 1, 6, 304, 512, 1, 0.25]]

#     model = EfficientNetV2(model_cnf=model_config,
#                            num_classes=num_classes,
#                            dropout_rate=0.3)
#     return model


# def efficientnetv2_l(num_classes: int = 10):
#     """
#     EfficientNetV2
#     https://arxiv.org/abs/2104.00298
#     """
#     # train_size: 384, eval_size: 480

#     # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
#     model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
#                     [7, 3, 2, 4, 32, 64, 0, 0],
#                     [7, 3, 2, 4, 64, 96, 0, 0],
#                     [10, 3, 2, 4, 96, 192, 1, 0.25],
#                     [19, 3, 1, 6, 192, 224, 1, 0.25],
#                     [25, 3, 2, 6, 224, 384, 1, 0.25],
#                     [7, 3, 1, 6, 384, 640, 1, 0.25]]

#     model = EfficientNetV2(model_cnf=model_config,
#                            num_classes=num_classes,
#                            dropout_rate=0.4)
#     return model



import torch
import torch.nn as nn
import math

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=10, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)
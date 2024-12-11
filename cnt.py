import torch
import torch.nn as nn
import math
import torchvision
import itertools

from timm.models.vision_transformer import trunc_normal_

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

class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                nn.SiLU(),
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
        nn.SiLU()
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
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
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
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m        

# class PatchMerging(torch.nn.Module):
#     def __init__(self, dim, out_dim, input_resolution):
#         super().__init__()
#         hid_dim = int(dim * 4)
#         self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
#         self.act = torch.nn.ReLU()
#         self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
#         self.se = SELayer(hid_dim,hid_dim)
#         self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

#     def forward(self, x):
#         x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
#         return x

class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim,resolution):
        super().__init__()
        self.cfgs = [[4, out_dim, 16//resolution, 2, 1],
                     [6, out_dim, 32//resolution, 1, 1]]
        input_channel = _make_divisible(out_dim , 8)
        layers = [conv_3x3_bn(dim,out_dim,1)]
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c , 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
    def forward(self, x):
        return self.features(x)

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.1):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h, resolution, drop=0.1):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = torch.nn.SiLU()
        self.drop = self.drop = nn.Dropout(drop)
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        x = self.pw2(self.drop(self.act(self.pw1(x))))
        return x


class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=4,
                 kernels=[5, 5, 5, 5],
                 attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.SiLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))
        self.attn_drop = nn.Dropout(attn_drop)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0: # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1) # BNN
            attn = self.attn_drop(attn)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=4,
                 window_resolution=4,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        
        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio, 
                                resolution=window_resolution,
                                kernels=kernels,)

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))
               
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x
    
class Patch_embed(torch.nn.Module):
    def __init__(self, embed_dim,width_mult=1.):
        super(Patch_embed, self).__init__()
        self.cfgs = [[1, 16, 2, 2, 0],
                     [2, 24, 2, 2, 0],
                     [4, 48, 4, 1, 1],
                     [4, embed_dim, 4, 1, 1]]
        input_channel = _make_divisible(embed_dim//8 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 1)]
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
    def forward(self, x):
        return self.features(x)

class EfficientViTBlock(torch.nn.Module):    
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=4,
                 window_resolution=4,
                 kernels=[3, 3, 3, 3],):
        super().__init__()
            
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
                    resolution=resolution, window_resolution=window_resolution, kernels=kernels))
                
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
    
class cnt(torch.nn.Module):
    def __init__(self, img_size=32,
                 patch_size=4,
                 num_classes=10,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 96, 128],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 1],
                 num_heads=[4, 4, 4],
                 window_size=[4, 4, 4],
                 kernels=[5, 3, 3, 1],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=True,):
        super().__init__()

        resolution = img_size
        # Patch embedding
        self.patch_embed = Patch_embed(embed_dim=embed_dim[0])

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        # Build EfficientViT blocks
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                eval('self.blocks' + str(i+1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                # Build EfficientViT downsample block
                #('Subsample' stride)
                blk = eval('self.blocks' + str(i+2))
                resolution_ = (resolution - 1) // do[1] + 1
                # blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                #                     Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)),))
                blk.append(PatchMerging(*embed_dim[i:i + 2],resolution))
                resolution = resolution_
                # blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                #                     Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)),))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)
        self.blocks4 = torch.nn.Sequential(nn.Conv2d(128, 1024, 2, padding=0),
                nn.BatchNorm2d(1024),
                nn.SiLU())
        # Classification head
        # self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        # self.distillation = distillation
        # if distillation:
        #     self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.head = BN_Linear(1024, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(1024, num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x
    
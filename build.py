import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientvit import EfficientViT
from timm.models.registry import register_model

EfficientViT_m0 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [64, 128, 192],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],
    }

EfficientViT_m1 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 144, 192],
        'depth': [1, 2, 3],
        'num_heads': [2, 3, 3],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m2 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 192, 224],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 2],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m3 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 240, 320],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],
    }

EfficientViT_m4 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 256, 384],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m5 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [192, 288, 384],
        'depth': [1, 3, 4],
        'num_heads': [3, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m32 = {
        'img_size': 32,
        'patch_size': 16,
        'embed_dim': [128, 192, 224],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [4, 4, 4],
        'kernels': [5, 3, 3, 1],
    }
@register_model
def EfficientViT_M0(num_classes=10, distillation=False, model_cfg=EfficientViT_m0):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

@register_model
def EfficientViT_M1(num_classes=10, distillation=False, model_cfg=EfficientViT_m1):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

@register_model
def EfficientViT_M2(num_classes=10, distillation=False, model_cfg=EfficientViT_m2):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

@register_model
def EfficientViT_M3(num_classes=10, distillation=False, model_cfg=EfficientViT_m3):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

@register_model
def EfficientViT_M4(num_classes=10, distillation=False, model_cfg=EfficientViT_m4):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

@register_model
def EfficientViT_M5(num_classes=10, distillation=True, model_cfg=EfficientViT_m5):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

@register_model
def EfficientViT_M32(num_classes=10, distillation=True, model_cfg=EfficientViT_m32):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

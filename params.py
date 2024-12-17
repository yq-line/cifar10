import torch
import torchvision
from ptflops import get_model_complexity_info
from model import *
from build import *
from fastervit import *
from cnt import cnt
from airbench96 import *
from clipres import ModifiedResNet
from swinv2 import SwinTransformerV2
from mambaout import *
# Model


# model = make_net(hyp['net'])
model = cnt()
# model = mambaout_kobe()
# model = ModifiedResNet([2,3,4,5],1024,128)
# model = SwinTransformerV2()
# model = torchvision.models.swin_t()

flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
print('flops: ', flops, 'params: ', params)


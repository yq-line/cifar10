import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import default_collate
from earlystoping import EarlyStopping
from math import cos,pi
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#model = torchvision.models.efficientnet_v2_m(weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT)
class Effnm(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.efficientnet_v2_m(weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT)
        num_ftrs = self.model.classifier[1].in_features 
        self.model.classifier[1] = nn.Linear(num_ftrs, 10)

    def forward(self,x):
        y = self.model(x)
        return y
    
model = Effnm()
for param in model.parameters(): 
    param.requires_grad = False
model.classifier[1].requires_grad = True
# for k,v in model.named_parameters():
#     v.requires_grad = False
#     #if k == 'cls.weight' or k=='cls.bias':
#     if k in 'cls':
#         v.requires_grad = True
#     print(k,v.requires_grad)
#print(*list(model.children()))
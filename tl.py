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

ROOT = './data'
def collate_fn(batch):
    cutmix = v2.CutMix(num_classes=10)
    mixup = v2.MixUp(num_classes=10)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    return cutmix_or_mixup(*default_collate(batch))

batch_size = 64
# normalization = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
img_size = [224, 224]
# train_transforms = transforms.Compose([
#     # transforms.RandomCrop(32, padding=4),
#     transforms.RandomResizedCrop(img_size[0]),
#     # transforms.RandomHorizontalFlip(),
#     torchvision.transforms.RandAugment(num_ops = 2, magnitude = 9, num_magnitude_bins = 31),
#     transforms.ToTensor(),
#     normalization
# ])

train_transforms = v2.Compose([
# v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.PILToTensor(),
    v2.RandomResizedCrop(img_size[0]),
    v2.RandAugment(num_ops = 2, magnitude = 9, num_magnitude_bins = 31),
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    v2.RandomErasing()
])
val_transforms = v2.Compose([
    v2.PILToTensor(),
    v2.Resize(img_size[1]),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])
# val_transforms = transforms.Compose([
#     transforms.Resize(img_size[1]),
#     transforms.ToTensor(),
#     normalization
# ])

train_set = torchvision.datasets.CIFAR10(ROOT, train=True, download=True,transform=train_transforms)
test_set = torchvision.datasets.CIFAR10(ROOT, train=False, download=True,transform=val_transforms)



train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

def warmup_cosine(optimizer, current_epoch, max_epoch, lr_min=0.0001, lr_max=0.1, warmup_epoch = 10):
    if current_epoch < warmup_epoch:
        lr = lr_min + (lr_max-lr_min) * current_epoch / warmup_epoch
    elif current_epoch < 260:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch - 240))) / 2
    else:
        lr = lr_min
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
def train(model, train_loader,valid_loader,epochs, criterion, optimizer,warmup_epoch,device,PATH):
    train_losses = []
    val_losses = []
    acces = []
    lr_max = 0.0001
    lr_min = 0.00001
    early_stopping = EarlyStopping( PATH)
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        train_total,val_total = 0 , 0
        correct = 0
        val_loss = 0.0
        warmup_cosine(optimizer=optimizer,current_epoch=epoch,max_epoch=epochs,lr_min=lr_min,lr_max=lr_max,warmup_epoch=warmup_epoch)
        print(optimizer.param_groups[0]['lr'])
        for i,data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(inputs.half(),outputs, labels)
            train_loss += loss.item()
            # _, predicted = torch.max(outputs.data, 1)
            train_total += 1
            # train_total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss / train_total:.3f}')
        # print statisticss
        print(f'[{epoch + 1}] loss: {train_loss / train_total:.3f}')
        train_losses.append(train_loss / train_total)
        train_loss,train_total = 0,0
        model.eval()
        for data in valid_loader:
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(inputs.half(),outputs, labels)
            # loss.backward()
            # optimizer.step()
            val_loss += loss.item()
            val_total += labels.size(0)
            correct += (predicted == labels).sum().item()

        model.train()
        print(f'[epoch: {epoch + 1}] loss: {val_loss / len(valid_loader)} acc: {100 * correct / val_total} %')
        val_losses.append(val_loss / len(valid_loader))
        acc = 100 * correct / val_total
        acces.append(acc)
        # scheduler.step()
        early_stopping(acc, model)
        if early_stopping.early_stop:
            print('Early Stopping')
            return train_losses,val_losses,acces

        val_loss,val_total,correct = 0,0,0

    path = os.path.join(PATH, 'finally_network.pth')
    torch.save(model.state_dict(), path)
    return train_losses,val_losses,acces

model = torchvision.models.efficientnet_v2_m(weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT)
class Seresnet50(nn.Module):
    def __init__(self):
        super(Seresnet50,self).__init__()
        # self.model = models.resnet18(pretrained=True)
        
        self.model = nn.Sequential(*list(model.children())[:7]) # 获取模型的前7层
        print(len(list(model.children())))
    def forward(self,x):
        y = self.model(x)
        return y
    
net = Seresnet50()
# print(net)

# 设置是否需要更新梯度
for k,v in net.named_parameters():
    # v.requires_grad = True
    # k 是某一层模型名称，v为该层的模型参数
    if k == 'model.6.5.se.fc.2.weight':
        v.requires_grad = False
    print(k,v.requires_grad)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# epochs = 100
# warmup_epoch = 10
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.008, momentum=0.9, weight_decay=1E-4)
# PATH = './weights'
# train_losses,val_losses,acces = train(model,train_loader,val_loader,epochs,criterion,optimizer,warmup_epoch,device,PATH)
# plt.figure(figsize=(10, 7))
# # plt.plot(acces, color='green', label='train accuracy')
# plt.plot(val_losses, color='blue', label='val loss')
# plt.plot(train_losses, color='red', label='train loss')
# plt.xlabel('Epochs')
# plt.savefig(f"./tlresult.png")
# plt.show()
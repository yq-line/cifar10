import os
import torch
import torchvision
import torchvision.transforms as transforms

ROOT = './data'

def get_dataloader(train=True):
    batch_size = 32
    normalization = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalization
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalization
    ])

    train_set = torchvision.datasets.CIFAR10(ROOT, train=True, download=True,transform=train_transforms)
    val_set = torchvision.datasets.CIFAR10(ROOT, train=False, download=True,transform=val_transforms)

 

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    if train:
        return train_loader
    else:
        return val_loader

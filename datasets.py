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
def k_fold_get_dataloader(train=True,k=10):
    batch_size = 128
    normalization = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandAugment(num_ops = 2, magnitude = 3, num_magnitude_bins = 7),
        transforms.ToTensor(),
        normalization
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalization
    ])

    train_set = torchvision.datasets.CIFAR10(ROOT, train=True, download=True,transform=train_transforms)
    test_set = torchvision.datasets.CIFAR10(ROOT, train=False, download=True,transform=val_transforms)
    num_samples = len(train_set)
    fold_size = num_samples // k
    train_loader = []
    for fold in range(k):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size

        indices = list(range(start_idx, end_idx))
        sampler = torch.utils.data.SubsetRandomSampler(indices)

        data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=2)
        train_loader.append(data_loader)

    # train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    if train:
        return train_loader
    else:
        return test_loader

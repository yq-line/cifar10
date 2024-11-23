import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import default_collate
ROOT = './data'

def get_dataloader(train=True):
    batch_size = 128
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
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    val_transforms = v2.Compose([
        v2.PILToTensor(),
        v2.Resize(img_size[1]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    # val_transforms = transforms.Compose([
    #     transforms.Resize(img_size[1]),
    #     transforms.ToTensor(),
    #     normalization
    # ])

    train_set = torchvision.datasets.CIFAR10(ROOT, train=True, download=True,transform=train_transforms)
    test_set = torchvision.datasets.CIFAR10(ROOT, train=False, download=True,transform=val_transforms)

 

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    if train:
        return train_loader,val_loader
    else:
        return val_loader


def collate_fn(batch):
    cutmix = v2.CutMix(num_classes=10)
    mixup = v2.MixUp(num_classes=10)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    return cutmix_or_mixup(*default_collate(batch))


def k_fold_get_dataloader(train=True,k=10):
    batch_size = 128
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
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    val_transforms = v2.Compose([
        v2.PILToTensor(),
        v2.Resize(img_size[1]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    # val_transforms = transforms.Compose([
    #     transforms.Resize(img_size[1]),
    #     transforms.ToTensor(),
    #     normalization
    # ])

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

        data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=2,collate_fn=collate_fn)
        train_loader.append(data_loader)

    # train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    if train:
        return train_loader
    else:
        return test_loader

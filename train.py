from model import *
from datasets import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from earlystoping import EarlyStopping
from build import *
from math import cos, pi
# from swinv2 import *
# from manbavision import *
from cnt import cnt
from airbench96 import *
from eval import load_weights
from losses import DistillationLoss

# def train(model, train_loader,epochs, criterion, optimizer,device,PATH,k):
#     train_losses = []
#     val_losses = []
#     acces = []
#     early_stopping = EarlyStopping( PATH)
#     for epoch in range(epochs):  # loop over the dataset multiple times
#         train_loss = 0.0
#         train_total,val_total = 0 , 0
#         correct = 0
#         val_loss = 0.0
#         for fold in range(k):
#             train_loaders = train_loader[:fold] + train_loader[fold+1:]
#             valid_loader = train_loader[fold]
#             for train_data in train_loaders:
#                 for i,data in enumerate(train_data, 0):
#                     inputs, labels = data[0].to(device), data[1].to(device)

#                     # zero the parameter gradients
#                     optimizer.zero_grad()

#                     # forward + backward + optimize
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#                     train_loss += loss.item()
#                     _, predicted = torch.max(outputs.data, 1)
#                     train_total += 1
#                     # train_total += labels.size(0)
#                     # correct += (predicted == labels).sum().item()
#                     loss.backward()
#                     optimizer.step()

#                     # print statistics
#                 print(f'[{epoch + 1}, {fold + 1}, {i + 1:5d}] loss: {train_loss / train_total:.3f}')
#             train_losses.append(train_loss / train_total)
#             train_loss,train_total = 0,0
#             for data in valid_loader:
#                 images, labels = data[0].to(device), data[1].to(device)
#                 optimizer.zero_grad()
#                 # calculate outputs by running images through the network
#                 outputs = model(images)
#                 # the class with the highest energy is what we choose as prediction
#                 _, predicted = torch.max(outputs.data, 1)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 val_loss += loss.item()
#                 val_total += labels.size(0)
#                 correct += (predicted == torch.argmax(labels)).sum().item()

#             print(f'[epoch: {epoch + 1}, fold: {fold + 1}] loss: {val_loss / len(valid_loader)} acc: {100 * correct / val_total} %')
#             val_losses.append(val_loss / len(valid_loader))
#             acces.append(100 * correct / val_total)

#             early_stopping(val_loss, model)
#             if early_stopping.early_stop:
#                 print('Early Stopping')
#                 return train_losses,val_losses,acces

#             val_loss,val_total,correct = 0,0,0

#     path = os.path.join(PATH, 'finally_network.pth')
#     torch.save(model.state_dict(), path)
#     return train_losses,val_losses,acces

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
    lr_max = 0.01
    lr_min = 0.00015
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

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # k = 10
    # train_loader = get_dataloader()
    train_loader,val_loader = get_dataloader(train=True)
    model = cnt()
    # net = EfficientViT_M0()
    # net = faster_vit_0_224()
    model.to(device)
    epochs = 500
    warmup_epoch = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.008, momentum=0.9, weight_decay=1E-4)
    teachermodel = model_trainbias = make_net(hyp['net'])
    weights = load_weights('./weights/1.pth')
    teachermodel.load_state_dict(weights)
    teachermodel.to(device)
    teachermodel.eval()
    criterion = DistillationLoss(criterion, teachermodel, 'soft', 0.4, 3.0)
    # milestones = [60,110,200,300]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.25)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=1e-4)
    PATH = './weights'
    train_losses,val_losses,acces = train(model,train_loader,val_loader,epochs,criterion,optimizer,warmup_epoch,device,PATH)
    plt.figure(figsize=(10, 7))
    # plt.plot(acces, color='green', label='train accuracy')
    plt.plot(val_losses, color='blue', label='val loss')
    plt.plot(train_losses, color='red', label='train loss')
    plt.xlabel('Epochs')
    plt.savefig(f"./result.png")
    plt.show()
if __name__ == '__main__':
    main()

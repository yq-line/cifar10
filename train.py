from model import *
from datasets import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from earlystoping import EarlyStopping


def train(model, train_loader,epochs, criterion, optimizer,device,PATH,k):
    train_losses = []
    val_losses = []
    acces = []
    early_stopping = EarlyStopping( PATH)
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        train_total,val_total = 0 , 0
        correct = 0
        val_loss = 0.0
        for fold in range(k):
            train_loaders = train_loader[:fold] + train_loader[fold+1:]
            valid_loader = train_loader[fold]
            for train_data in train_loaders:
                for i,data in enumerate(train_data, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += 1
                    # train_total += labels.size(0)
                    # correct += (predicted == labels).sum().item()
                    loss.backward()
                    optimizer.step()

                    # print statistics
                print(f'[{epoch + 1}, {fold + 1}, {i + 1:5d}] loss: {train_loss / train_total:.3f}')
            train_losses.append(train_loss / train_total)
            train_loss,train_total = 0,0
            for data in valid_loader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'[epoch: {epoch + 1}, fold: {fold + 1}] loss: {val_loss / len(valid_loader)} acc: {100 * correct / val_total} %')
            val_losses.append(val_loss / len(valid_loader))
            acces.append(100 * correct / val_total)

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print('Early Stopping')
                return train_losses,val_losses,acces

            val_loss,val_total,correct = 0,0,0

    path = os.path.join(PATH, 'finally_network.pth')
    torch.save(model.state_dict(), path)
    return train_losses,val_losses,acces


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    k = 10
    # train_loader = get_dataloader()
    train_loader = k_fold_get_dataloader(train=True,k=k)
    net = efficientnetv2_s()
    net.to(device)
    epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=1E-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=1e-4)
    PATH = './weights'
    train_losses,val_losses,acces = train(net,train_loader,epochs,criterion,optimizer,device,PATH,k)
    plt.figure(figsize=(10, 7))
    # plt.plot(acces, color='green', label='train accuracy')
    plt.plot(val_losses, color='blue', label='val loss')
    plt.plot(train_losses, color='red', label='train loss')
    plt.xlabel('Epochs')
    plt.savefig(f"./result.png")
    plt.show()
if __name__ == '__main__':
    main()

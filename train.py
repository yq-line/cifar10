from model import *
from datasets import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib
import matplotlib.pyplot as plt

# def k_fold_train(model, train_loader,epochs, criterion, optimizer,device,k):
#     k_fold_scores = []
#     data_size = len(train_loader)

#     # 计算每个fold的数据量
#     fold_size = data_size // k

#     # 对于K值和数据集中每个fold
#     for fold_idx in range(k):
#         # 将数据集分成训练集和测试集
#         start_idx = fold_idx * fold_size
#         end_idx = start_idx + fold_size

#         validation_data = train_loader[start_idx:end_idx]
#         training_data = torch.cat((train_loader[:start_idx], train_loader[end_idx:]), dim=0)

#         # 创建训练器
#         trainer = create_trainer(model, learning_rate)

#         # 训练模型
#         train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
#         for epoch in range(num_epochs):
#             # 训练模型
#             for batch_index, (x_data, y_data) in enumerate(train_loader):
#                 trainer.train_step(x_data, y_data)

#         # 运行测试数据
#         validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
#         accuracy = evaluate_accuracy(model, validation_loader)
#         print(f"Fold {fold_idx+1} accuracy: {accuracy}")

#         # 保存验证分数
#         k_fold_scores.append(accuracy)

#     # 返回平均验证分数
#     return sum(k_fold_scores) / k

def train(model, train_loader,epochs, criterion, optimizer,device,PATH):
    train_losses = []
    train_acces = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 250 == 249:    # print every 250 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 250:.3f} acc:{100 * correct // total} %')
                # total = 0
                # correct = 0
                # running_loss = 0.0
        print(f'[epoch: {epoch + 1}] loss: {running_loss / len(train_loader)} acc: {100 * correct // total} %')
        train_losses.append(running_loss / len(train_loader))
        train_acces.append(correct / total)
        total = 0
        correct = 0
        running_loss = 0.0
    torch.save(model.state_dict(), PATH)
    return train_losses,train_acces


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader = get_dataloader()
    net = effnetv2_s()
    net.to(device)
    epochs = 3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    PATH = './weights/cifar_net.pth'
    k = 10
    train_losses,train_acces = train(net,train_loader,epochs,criterion,optimizer,device,PATH)
    plt.figure(figsize=(10, 7))
    plt.plot(train_acces, color='green', label='train accuracy')
    plt.plot(train_losses, color='red', label='train loss')
    plt.xlabel('Epochs')
    plt.savefig(f"./result.png")
    plt.show()
if __name__ == '__main__':
    main()

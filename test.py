from datasets import *
from model import *
import torch.nn as nn
import os
from build import *
from airbench96 import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_weights(path):
    # torch2.0版本以上支持torch.compile来跑模型，会快，但是compile还只支持linux系统
    # compile后的模型存权重的时候层的名字前面会加上'_orig_mod.'
    # 这段代码就是把这个删掉
    # 传入模型的路径（.pth），返回权重，直接使用model.load_state_dict(weight)就能读进去
    weight = torch.load(path)
    new_weight = weight.copy()
    keys_list = list(weight.keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            del_key = key.replace('_orig_mod.', '')
            new_weight[del_key] = weight[key]
            del new_weight[key]
    return new_weight

def plot_confusion_matrix(cm, classes,save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)  # 保存为图片文件
    plt.show()

def eval(model, test_loader,device,class_names):
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
    cm = confusion_matrix(all_labels, all_preds)
    save_path = 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names,save_path)

def test(test_loader, model,device,class_names):
    model.eval()
    total = 0
    correct = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100. * correct / total
    print(f'Test Acc: {accuracy:.2f}%')

    # 计算并绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    save_path = 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names,save_path)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.efficientnet_v2_m()
    num_ftrs = model.classifier[1].in_features 
    model.classifier[1] = nn.Linear(num_ftrs, 10)
    #weight = torch.load('./weights/tlmbest_network.pth',weights_only=True)
    model.load_state_dict(torch.load('./weights/tlmbest_network.pth',weights_only=True,map_location='cuda:0'))
    model.to(device)
    # model.load_state_dict(torch.load(PATH, weights_only=True,map_location='cuda:0'))
    # prepare dataset
    test_loader = get_dataloader(train=False)
    # test
    # eval(model, test_loader,device)
    # CIFAR-10 类别名称
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    test(test_loader,model,device,class_names)


if __name__ == '__main__':
    main()
    
from datasets import *
from model import *
import torch.nn as nn
import os
from build import *
from airbench96 import *

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

def eval(model, test_loader,device):
    correct = 0
    total = 0
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

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
def test(test_loader, model,device,n_classes):
    model.eval()
    # test_loss = 0 
    target_num = torch.zeros((1, n_classes)) # n_classes为分类任务类别数量
    predict_num = torch.zeros((1, n_classes))
    acc_num = torch.zeros((1, n_classes))

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.half()
            outputs = model(inputs)
            # loss = criterion(outputs, targets)

            # test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
            tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)  # 得到数据中每类的数量
            acc_mask = pre_mask * tar_mask 
            acc_num += acc_mask.sum(0) # 得到各类别分类正确的样本数量

        recall = acc_num / target_num
        precision = acc_num / predict_num
        F1 = 2 * recall * precision / (recall + precision)
        accuracy = 100. * acc_num.sum(1) / target_num.sum(1)

        print('Test Acc {}, recal {}, precision {}, F1-score {}'.format(accuracy, recall, precision, F1))
        return 

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = make_net(hyp['net'])
    # load model
    # model = nn.DataParallel(model)
    PATH = './weights/1.pth'
    weights = load_weights(PATH)
    model.load_state_dict(weights)
    model.to(device)
    # model.load_state_dict(torch.load(PATH, weights_only=True,map_location='cuda:0'))
    # prepare dataset
    test_loader = get_dataloader(train=False)
    # test
    # eval(model, test_loader,device)
    nc=10
    test(test_loader,model,device,nc)



if __name__ == '__main__':
    main()
    
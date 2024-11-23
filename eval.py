from datasets import *
from model import *
import torch.nn as nn
import os
from build import *

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
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EfficientViT_M5()
    model.to(device)
    # load model
    PATH = './weights/best_network.pth'
    model.load_state_dict(torch.load(PATH, weights_only=True))
    # prepare dataset
    test_loader = get_dataloader(train=False)
    # test
    eval(model, test_loader,device)



if __name__ == '__main__':
    main()
    
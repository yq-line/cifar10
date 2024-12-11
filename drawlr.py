from math import cos, pi
import matplotlib.pyplot as plt

def warmup_cosine(current_epoch, max_epoch, lr_min=0.00015, lr_max=0.01, warmup_epoch = 10):
    if current_epoch < warmup_epoch:
        lr = lr_min + (lr_max-lr_min) * current_epoch / warmup_epoch
    elif current_epoch < 260:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch - 240))) / 2
    else:
        lr = lr_min
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr
    return lr
def main():
    epochs = 500
    lr = []
    for epoch in range(epochs):
        lr.append(warmup_cosine(current_epoch=epoch,max_epoch=epochs))
    plt.figure(figsize=(10, 7))
    plt.plot(lr, color='blue', label='learn rate')
    plt.xlabel('Epochs')
    plt.savefig(f"./draw.png")
    plt.show()
if __name__ == '__main__':
    main()
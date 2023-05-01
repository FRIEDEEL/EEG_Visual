# Imports
# ------------------------- I am a split line \(ow <) -------------------------
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim
import torch.nn.functional as func
# ------------------------- I am a split line \(ow <) -------------------------
import numpy as np
import matplotlib.pyplot as plt
# ------------------------- I am a split line \(ow <) -------------------------
import copy
import json
import time
import calendar
import os
# ------------------------- I am a split line \(ow <) -------------------------
from utils import *
import models
# ------------------------- I am a split line \(ow <) -------------------------
import CSwork as cs
# ------------------------- I am a split line \(ow <) -------------------------

# 在cuda上训练
TO_CUDA=True
if TO_CUDA:
    if not torch.cuda.is_available():
        TO_CUDA=False
        print("Warning: Cuda is not available. Using cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hypoparameters
BATCH_SIZE = 256
EPOCH = 1000
# Learning rate and decay. The learning rate #1 decays by #2 every #3 epochs.
LEARNING_RATE = 0.1  # 1
LR_DECAY_RATE = 0.1 # 2, float in [0,1]
LR_DECAY_FREQ = 100  # 3, int
WEIGHT_DECAY = 0.005 # 正则化项的Lambda
HYPERPARA_INFO = {  # 将超参数记录为字典，为了存储方便
    "optimizer": "SGD",
    "lr": LEARNING_RATE,
    "batch size": BATCH_SIZE,
    "epochs": EPOCH,
    "learning rate decay rate": LR_DECAY_RATE,
    "learning rate decay period": LR_DECAY_FREQ,
    "regularization weight decay(lambda)": WEIGHT_DECAY
}
# 存储训练信息的log文件
LOGFILE = 0
# 数据的根目录
DATAPATH="../data/"


def _gen_logfile_pth():
    global LOGFILE
    _t = time.strftime("%y%m%d_%H%M", time.localtime())
    name = "training_logs/{}".format(_t)
    if os.path.exists(name + ".txt"):
        for i in range(10):
            newname = name + "_{}".format(i + 1)
            if not os.path.exists(newname + ".txt"):
                LOGFILE = newname + ".txt"
                break
    else:
        LOGFILE = name + ".txt"

_gen_logfile_pth()

# print(LOGFILE)

def main():
    # test()
    train(datadir="../data/processed/3",
          save_to_log=True)
    # data_test()
    # train_on_CSdataset()
    pass

# 用CS的数据集跑一下
def train_on_CSdataset():

    # 实例化模型
    model = models.LSTMModel(lstm_input_size=128,
                             lstm_size=128,
                             lstm_output_size=128)
    # 实例化optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
    # 实例化loader
    train_loader = cs.loaders["train"]
    val_loader = cs.loaders["val"]
    # 初始化训练信息
    losses = []
    accuracies = []
    for epoch in range(EPOCH):
        # 更新SGD学习率
        lr = LEARNING_RATE * (LR_DECAY_RATE)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 训练
        TL, TA = train_in_epoch(model, train_loader, optimizer)
        # 验证
        VL, VA = eval_in_epoch(model, val_loader)
        # 输出训练信息
        print("epoch {},".format(epoch), end="\t")
        print("TL: {:.5f},\tTA: {:.5f}".format(TL, TA), end="\t")
        print("VL: {:.5f},\tVA: {:.5f}".format(VL, VA), end="")
        print("")

        # 记录训练信息
        losses.append([TL, VL])
        accuracies.append([TA, VA])
    pass

# 细分版本的训练代码入口
def train(datadir, save_to_log=True):
    """Entrance to training code.

    Args:
        datadir (str): the directory where data is stored.
        save_to_log (bool, optional): If `True`, save the training info (loss, \
            accuracy, etc, to a `.txt` file named with timestamp). \
            Defaults to True.

    Returns:
        tuple : (loss, accuracy) info, both stored in a list.
    """    
    if save_to_log:
        with open(LOGFILE,"a+") as f:
            f.write("Hypoparas:\n")
            for key in HYPERPARA_INFO:
                f.write("{} : {}\n".format(key,HYPERPARA_INFO[key]))
            f.write("\n")
    # 以下是一些准备工作
    # 准备数据
    dataset = load_all(datadir)
    # 划分训练、验证、测试集
    datasets = split_dataset(dataset, [0.7, 0.2, 0.1])
    train_subset = datasets[0]
    val_subset = datasets[1]
    test_subset = datasets[2]
    # 实例化模型
    model = models.LSTMModel()
    if TO_CUDA:
        model=model.to(device)
    # 实例化optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
    # 实例化loader
    train_loader = DataLoader(train_subset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)
    # 初始化训练信息
    losses = []
    accuracies = []
    # 准备工作到此结束

    for epoch in range(EPOCH):
        # 更新SGD学习率
        lr = LEARNING_RATE * (LR_DECAY_RATE)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 训练
        TL, TA = train_in_epoch(model, train_loader, optimizer)
        # 验证
        VL, VA = eval_in_epoch(model, val_loader)
        # 输出训练信息
        print("epoch {},".format(epoch+1), end="\t")
        print("TL: {:.5f},\tTA: {:.5f}".format(TL, TA), end="\t")
        print("VL: {:.5f},\tVA: {:.5f}".format(VL, VA), end="")
        print("")

        # 记录训练信息
        if save_to_log:
            with open(LOGFILE,"a") as f:
                f.write("Epoch:{},TL:{},VL:{},TA:{},VA:{}\n".format(epoch+1,TL,VL,TA,VA))
        losses.append([TL, VL])
        accuracies.append([TA, VA])

    return losses, accuracies

# 检查数据集是否出问题了
def data_test():
    # 准备数据
    dataset = load_file_to_dataset([
        'data/processed/norm2/230402_1.pth',
        'data/processed/norm2/230402_2.pth',
        'data/processed/norm2/230406_1.pth',
        'data/processed/norm2/230408_1.pth',
        'data/processed/norm2/230411_1.pth'
    ])
    # 划分训练、验证、测试集
    split_subsets = split_dataset(dataset, [0.7, 0.2, 0.1])
    train_subset = split_subsets[0]
    val_subset = split_subsets[1]
    test_subset = split_subsets[2]

    # 随便挑一组东西出来作图看看
    item = train_subset[15]
    print(item[0].shape)
    print(len(dataset))
    print(torch.mean(item[0]))
    print(torch.std(item[0]))
    fig = plt.figure(figsize=[10, 8])
    for i in range(16):
        ax = fig.add_subplot(16, 1, i + 1)
        ax.plot(item[0][:, i])
    print(item[0]["label"])
    plt.show()

def train_in_epoch(model, loader, optimizer):
    # 初始化记录
    model.train()
    batch_count = 0  # batch计数
    epoch_loss = 0.0  # 总loss
    epoch_accuracy = 0.0  # epoch中预测准确率之和

    for idx, (input, target) in enumerate(loader):
        if TO_CUDA:
            input=input.to(device)
            target=target.to(device)
        output = model(input)
        loss = func.cross_entropy(output, target)  # 交叉熵损失函数
        epoch_loss += loss.item()  # 计算总损失
        # 评估
        _, pred = output.data.max(1)  # 给出预测分类
        correct = pred.eq(target.data).sum().item()  # 正确分类计数
        accuracy = correct / input.data.size(0)  # 计算分类准确率
        epoch_accuracy += accuracy  # 累计准确率，用于平均
        batch_count += 1  # batch的个数
        # 反向传播过程
        optimizer.zero_grad()  # 清除梯度信息
        loss.backward()  # 反向计算梯度
        optimizer.step()  # 根据梯度更新参数

    # 更新信息
    epoch_loss = epoch_loss / batch_count
    epoch_accuracy = epoch_accuracy / batch_count
    return epoch_loss, epoch_accuracy

def eval_in_epoch(model, loader):
    model.eval()
    torch.set_grad_enabled(True)
    # 初始化记录
    batch_count = 0  # batch计数
    epoch_loss = 0.0  # 总loss
    epoch_accuracy = 0.0  # epoch中预测准确率之和

    for idx, (input, target) in enumerate(loader):
        if TO_CUDA:
            input=input.to(device)
            target=target.to(device)
        output = model(input)
        loss = func.cross_entropy(output, target)  # 交叉熵损失函数
        epoch_loss += loss.item()  # 计算总损失
        # 评估
        _, pred = output.data.max(1)  # 给出预测分类
        correct = pred.eq(target.data).sum().item()  # 正确分类计数
        accuracy = correct / input.data.size(0)  # 计算分类准确率
        epoch_accuracy += accuracy  # 累计准确率，用于平均
        batch_count += 1  # batch的个数

    # 更新信息
    epoch_loss = epoch_loss / batch_count
    epoch_accuracy = epoch_accuracy / batch_count
    return epoch_loss, epoch_accuracy

if __name__ == '__main__':
    main()
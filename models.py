# imports
import numpy as np
# ------------------------- I am a split line \(ow <) -------------------------
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils , datasets , transforms
from torch.autograd import Variable
import torch.nn.functional as func

# initializing weights.
def init_weights(model):
    if isinstance(model,nn.LSTM):
        for para in model.parameters():
            try:
                torch.nn.init.xavier_normal_(para)
            except ValueError:
                pass

class LSTMModel(nn.Module):
    def __init__(self,
                 lstm_input_size=16,
                 lstm_size=16,
                 lstm_layers=1,
                 lstm_output_size=16,
                 classes=40):
        super().__init__()
        self.input_size = lstm_input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = lstm_output_size

        # LSTM层
        self.lstm = nn.LSTM(self.input_size,
                            self.lstm_size,
                            num_layers=self.lstm_layers,
                            batch_first=True)
        # 输出层，全连接
        self.output = nn.Linear(self.lstm_size, self.output_size)
        # batch归一化层
        self.bn = nn.BatchNorm1d(self.output_size)
        # 分类器，也是全连接线性层
        self.classifier = nn.Linear(lstm_output_size, classes)
        self.apply(init_weights)

    def forward(self,x):

        x=self.lstm(x)[0][:,-1,:]

        x=func.relu(self.output(x))

        x=self.bn(x)

        x=self.classifier(x)
        return x
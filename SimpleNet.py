import torch
from torch import nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputSize = 56
        self.hiddenSize = self.inputSize * 2 + 1
        self.outputSize = 3

        self.inputLayer = nn.Linear(self.inputSize, self.hiddenSize, bias=True)
        self.hiddenLayer = nn.Linear(self.hiddenSize, self.hiddenSize, bias=True)
        self.outputLayer = nn.Linear(self.hiddenSize, self.outputSize, bias=True)

    def forward(self, inputs):
        x = F.sigmoid(self.inputLayer(inputs))
        x = F.sigmoid(self.hiddenLayer(x))
        x = F.softmax(self.outputLayer(x), dim=0)
        return x
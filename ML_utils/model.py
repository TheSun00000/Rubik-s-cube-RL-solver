import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    """
    Linear layer with ReLU and BatchNorm
    """
    def __init__(self, input_prev, embed_dim):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(input_prev, embed_dim)
        self.elu = nn.ELU()
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, inputs):
        x = inputs
        x = self.fc(x)
        x = self.elu(x)
        x = self.bn(x)
        return x
        

class ResidualBlock(nn.Module):
    """
    Residual block with two linear layers
    """
    def __init__(self, embed_dim):
        super(ResidualBlock, self).__init__()
        self.linearblock_1 = LinearBlock(embed_dim, embed_dim)
        self.linearblock_2 = LinearBlock(embed_dim, embed_dim)

    def forward(self, inputs):
        x = inputs
        x = self.linearblock_1(x)
        x = self.linearblock_2(x)
        x += inputs # skip-connection
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.one_hot = nn.functional.one_hot
        self.Stack = nn.Sequential(
            LinearBlock(3*3*6*6, 5000),
            LinearBlock(5000, 1000),
            ResidualBlock(1000),
            ResidualBlock(1000),
            ResidualBlock(1000),
            ResidualBlock(1000),
        )
        self.Prediction = nn.Linear(1000, 18)

    def forward(self, inputs):
        x = inputs
        x = nn.functional.one_hot(x, num_classes=6).reshape(-1, 3*3*6*6).to(torch.float)
        x = self.Stack(x)
        logits = self.Prediction(x)
        return logits
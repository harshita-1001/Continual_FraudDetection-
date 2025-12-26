import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.fc1 = nn.Linear(features, features)
        self.fc2 = nn.Linear(features, features)

    def forward(self, x):
        residual = x
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        return torch.relu(out + residual)

class FraudResNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = self.res1(x)
        x = self.res2(x)
        return torch.sigmoid(self.out(x))

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mlp', 'mlpbn', 'mlpdrop']

class MLP(nn.Module):
    def __init__(self, units=400, num_classes=10):
        super(type(self), self).__init__()
        self.fc1 = nn.Linear(784, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLPBN(nn.Module):
    def __init__(self, units=400, num_classes=10):
        super(type(self), self).__init__()
        self.fc1 = nn.Linear(784, units)
        self.bn1 = nn.BatchNorm1d(units)
        self.fc2 = nn.Linear(units, units)
        self.bn2 = nn.BatchNorm1d(units)
        self.fc3 = nn.Linear(units, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class MLPDROP(nn.Module):
    def __init__(self, units=400, num_classes=10, drop=0.5):
        super(type(self), self).__init__()
        self.fc1 = nn.Linear(784, units)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def mlp(**kwargs):
    r"""3 Layer MLP model architecture
    """
    model = MLP(**kwargs)
    return model

def mlpbn(**kwargs):
    r"""3 Layer MLP model architecture
    """
    model = MLPBN(**kwargs)
    return model

def mlpdrop(**kwargs):
    r"""3 Layer MLP model architecture
    """
    model = MLPDROP(**kwargs)
    return model
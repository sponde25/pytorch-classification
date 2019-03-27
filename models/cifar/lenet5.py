import torch.nn as nn
import torch.nn.functional as F

__all__ = ['lenet5', 'lenet5bn', 'lenet5drop', 'lenet5bndrop']

class LeNet5(nn.Module):
    def __init__(self, input_channels = 3, dims = 32, num_classes=10):
        super(type(self), self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            #nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.fc1 = nn.Linear(16*dims*dims, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc(out)
        return out

class LeNet5Drop(nn.Module):
    def __init__(self, input_channels = 3, dims = 32, num_classes=10, drop=0.5):
        super(type(self), self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            nn.ReLU(),
            nn.Dropout2d(drop),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.Dropout2d(drop),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.fc1 = nn.Linear(16*dims*dims, 120)
        self.dp = nn.Dropout(drop)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = self.dp(F.relu(self.fc1(out)))
        out = self.dp(F.relu(self.fc2(out)))
        out = self.fc(out)
        return out

class LeNet5BN(nn.Module):
    def __init__(self, input_channels = 3, dims = 32, num_classes=10):
        super(type(self), self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.fc1 = nn.Linear(16*dims*dims, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc(out)
        return out

class LeNet5BNDrop(nn.Module):
    def __init__(self, input_channels = 3, dims = 32, num_classes=10):
        super(type(self), self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.fc1 = nn.Linear(16*dims*dims, 120)
        self.dp = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = self.dp(F.relu(self.bn1(self.fc1(out))))
        out = self.dp(F.relu(self.bn2(self.fc2(out))))
        out = self.fc(out)
        return out


def lenet5bn(**kwargs):
    r"""LeNet-5 model architecture
    """
    model = LeNet5BN(**kwargs)
    return model

def lenet5(**kwargs):
    return LeNet5(**kwargs)

def lenet5drop(**kwargs):
    return LeNet5Drop(**kwargs)

def lenet5bndrop(**kwargs):
    return LeNet5BNDrop(**kwargs)

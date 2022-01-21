import torch.nn as nn
import torch.nn.functional as F

__all__ = ['dsads',]

class ConvNet(nn.Module):
    def __init__(self, num_classes=19):
        super(ConvNet, self).__init__()
        final_kernel_size = [45, 6]
        self.conv1 = nn.Conv2d(1, 4, kernel_size= [1, 1], stride=[1,1], padding=[0, 0])
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 16, kernel_size= [1, 1], stride=[1, 1], padding=[0, 0])
        self.bn2= nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 256, kernel_size= final_kernel_size, stride=[1, 1], padding=[0, 0])
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256, 256)
        self.bn1_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2_fc = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn_fc3 = nn.BatchNorm1d(num_classes)

    def forward(self, x, feature=False):
        x = x.float()
        x = x.view(-1, 1, x.size(-2), x.size(-1)) # Batch * Channel * Height * Width
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = F.relu6(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu6(self.bn1_fc(self.fc1(x)))
        x = F.relu6(self.bn2_fc(self.fc2(x)))
        if feature:
            x1 = self.fc3(x)
            return x, x1  # x is feature, while x1 is the classifier
        x = self.fc3(x)
        return x


def dsads(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ConvNet(**kwargs)
    return model


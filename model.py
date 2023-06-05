import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiTaskResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiTaskResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.fc = nn.Linear(num_ftrs * 2 * 2, num_ftrs)
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc1 = nn.Linear(num_ftrs, 11)
        self.fc2 = nn.Linear(num_ftrs, 11)
        self.fc3 = nn.Linear(num_ftrs, 1)
        self.fc4 = nn.Linear(num_ftrs, 1)
        self.fc5 = nn.Linear(num_ftrs, 1)
        self.relu = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x))
        
        x3 = self.relu(self.fc3(x))
        
        x4 = self.relu(self.fc4(x))
        
        x5 = self.relu(self.fc5(x))
        
        return x1, x2, x3, x4, x5
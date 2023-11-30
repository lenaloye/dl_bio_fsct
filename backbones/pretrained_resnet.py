from torch import nn as nn
from torchvision import models

class PreTrainedResNet(nn.Module):
    def __init__(self, type='ResNet18', flatten = False):
        super(PreTrainedResNet, self).__init__()
        if type == 'ResNet18':
            resnet = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        elif type == 'ResNet34':
            resnet = models.resnet34(weigths = models.ResNet34_Weights.DEFAULT)

        trunk = [*resnet.children()][:-2]
        
        if flatten:
            trunk.append(nn.Flatten())
            self.final_feat_dim = 512*7*7
        else:
            self.final_feat_dim = [512, 7, 7]

        self.model = nn.Sequential(*trunk)
        
    def forward(self, x):
        out = self.model(x)
        return out

def PTResNet18(x_dim, flatten=True):
    return PreTrainedResNet('ResNet18', flatten)

def PTResNet34(x_dim, flatten=True):
    return PreTrainedResNet('ResNet34', flatten)

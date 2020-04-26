import torch.nn as nn
from efficientnet_pytorch import utils
from efficientnet_pytorch.model import EfficientNet

"""
EfficientNet reference: 

https://github.com/lukemelas/EfficientNet-PyTorch

"""

class RecognizeModel(nn.Module):
    def __init__(self):
        """
        :param size: 输入尺寸大小，为整数
        """
        super().__init__()
        self.layer1 = EfficientNet.from_pretrained('efficientnet-b7')
        self.linear = nn.Linear(1000, 4)

    def forward(self, x):
        """
        前向传播过程
        :return:
        """
        feat = self.layer1(x)
        result = self.linear(feat)
        return result


import torch.nn as nn
import torchvision.models as models


class RecognizeModel(nn.Module):
    def __init__(self):
        """
        :param size: 输入尺寸大小，为整数
        """
        super().__init__()
        self.layer1 = models.resnet50(pretrained=True)
        self.linear = nn.Linear(1000, 4)

    def forward(self, x):
        """
        前向传播过程
        :return:
        """
        feat = self.layer1(x)
        result = self.linear(feat)
        return result


# author time:2024-02-07
from torch.nn import functional as F
from .utils import *
import abc
import torchvision
from torch import nn, Tensor
from typing import Tuple, Sequence


"""
vgg的模型
"""
class BaseArch(nn.Module, metaclass=abc.ABCMeta):
    """图片特征抽取网络父 module。任何一个被用来作为特征抽取的预训练网络都应该继承此类，
    然后实现 extract_features(x) 抽象方法。
    """

    def __init__(self):
        super(BaseArch, self).__init__()

    def forward(self, x: Tensor) -> Tuple[list, list]:
        return self.extract_features(x)

    @abc.abstractmethod
    def extract_features(self, x: Tensor) -> Tuple[list, list]:
        """图片的特征抽取。对给定的图片 Tensor 计算并返回对应的 "内容特征" 和 "风格特征"
        """
        pass



class VGG(BaseArch, abc.ABC):
    """VGG 系列特征抽取网络

    Args:
        content_layers (Sequence): 需要进行内容特征（content features）提取的层序列
        style_layers (Sequence): 需要进行风格特征（style features）提取的层序列
    """

    def __init__(self, content_layers: Sequence, style_layers: Sequence):
        super(VGG, self).__init__()
        # print(style_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers

    @abc.abstractmethod
    def features_layer(self):
        """返回预训练 VGG 的特征层 module"""
        pass

    def extract_features(self, x: Tensor) -> Tuple[list, list]:
        content_features, style_features = [], []
        for i in range(len(self.features_layer())):
            x = self.features[i](x)
            # print(self.style_layers,'zzz')
            if str(i) in self.style_layers:
                style_features.append(x)
            if str(i) in self.content_layers:
                content_features.append(x)

        return content_features, style_features



class VGG16(VGG, abc.ABC):
    def __init__(self, content_layers: Sequence, style_layers: Sequence):
        super(VGG16, self).__init__(content_layers, style_layers)
        pretrained_net = torchvision.models.vgg16(weights=True)

        # 冻结特征层参数
        self.features = pretrained_net.features
        for param in self.features.parameters():
            param.requires_grad = False

    def features_layer(self):
        return self.features

class VGG19(VGG, abc.ABC):
    def __init__(self, content_layers: Sequence, style_layers: Sequence):
        super(VGG19, self).__init__(content_layers, style_layers)
        pretrained_net = torchvision.models.vgg19(pretrained=True)

        # 冻结特征层参数
        self.features = pretrained_net.features
        for param in self.features.parameters():
            param.requires_grad = False

    def features_layer(self):
        return self.features


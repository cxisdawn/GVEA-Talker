# author time:2024-02-07
from .utils import *
from torch import nn, Tensor

"""
关于风格损失与特征损失的函数
"""

class ContentLoss(nn.Module):
    """图片内容损失函数

    计算生成图片和内容图片对应内容特征层的均方误差（MSE），该损失函数是为了保证生成图片内容尽量接近内容图片的内容。
    ref: https://courses.d2l.ai/zh-v2/assets/notebooks/chapter_computer-vision/neural-style.slides.html#/
    """

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, batch_y: Tensor, batch_y_hat: Tensor) -> Tensor:
        content_loss = 0.0
        for y, y_hat in zip(batch_y, batch_y_hat):
            assert y.shape == y_hat.shape, "原内容风格层特征与 Transformer 网络输出对应层的内容风格特征形状不一致"
            content_loss += F.mse_loss(y, y_hat)
        return self.weight * content_loss


class StyleLoss(nn.Module):
    """图片风格损失函数

    通过特征映射的 gram 矩阵进行计算，该损失函数是为了保证在生成的图像中保风格图片的样式。
    ..ref: https://github.com/eriklindernoren/Fast-Neural-Style-Transfer/blob/master/train.py
    """

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, batch_y: Tensor, batch_gram_y_hat: Tensor) -> Tensor:
        style_loss = 0.0
        for y, gram_y_hat in zip(batch_y, batch_gram_y_hat):
            gram_y = gram_matrix(y)
            style_loss += F.mse_loss(gram_y, gram_y_hat[:batch_y[0].shape[0], :, :])
        return self.weight * style_loss


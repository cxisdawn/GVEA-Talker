import os
import torch
from torchvision import transforms
from .vgg_models import *
from collections import OrderedDict

# 模型训练输入数据的均值和标准差
RGB_MEAN, RGB_STD = Tensor([0.485, 0.456, 0.406]), Tensor([0.229, 0.224, 0.225])

def gram_matrix(batch_y: Tensor) -> Tensor:
    """计算批量图片 y 的 gram 矩阵值（计算风格损失时使用）

    .. ref: A Neural Algorithm of Artistic Style, http://arxiv.org/abs/1508.06576
    """
    (b, c, h, w) = batch_y.size()
    features = batch_y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    return features.bmm(features_t) / (c * h * w)

def content_image_transform(image_size: int,image_rgb) -> transforms.Compose:
    """预处理内容图片的 transform 定义
    image_rgb:[3,H,W]
    先将图片放大 1.15 倍，然后再随机剪裁出 image_size * image_size 大小的图片，然后转换为 Tensor
    最后使用 ImageNet 的 MEAN 和 STD 进行规范化。

    ..ref https://github.com/eriklindernoren/Fast-Neural-Style-Transfer/blob/master/utils.py
    """
    # return transforms.Compose([
    #     transforms.Resize(int(image_size * 1.15)), transforms.RandomCrop(image_size), transforms.ToTensor(),
    #     transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
    # ])
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
    ])
    transformed_out = transform(image_rgb)
    return transformed_out

def content_tensor_image_transform(image_size: int,tensor_rgb):
    """对输入数据是tensor的时候的处理 tensor_rgb[3,512,512]
        """
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将Tensor转换为PIL Image
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
    ])
    transformed_out = transform(tensor_rgb)
    return transformed_out
#
# def style_image_transform(image_size: int = None) -> transforms.Compose:
#     """预处理风格图片的 transform 定义
#
#     当不指定 image_size 参数（即为 None）时，风格图片保持原图大小；否则将被 resize 为 image_size * image_size 大小
#     其余预处理操作同 content_image_transform。
#     """
#     transform = [transforms.Resize((image_size, image_size))] if image_size else []
#     # noinspection PyTypeChecker
#     transform.extend([transforms.ToTensor(), transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)])
#     return transforms.Compose(transform)

# def denormalize_image(image: Tensor) -> Tensor:
#     """将张量图片使用 ImageNet 的 MEAN 和 STD 进行反规范化，还原为原来的 RGB 值
#
#     Args:
#         image (Tensor): 图片 shape 应为 (batch_size, channel, height, weight) 的 4d 张量
#     """
#     assert len(image.shape) == 4, "错误参数，被反规范化的图片应为一个 4d 张量"
#     for channel in range(image.shape[1]):
#         image[:, channel].mul_(RGB_STD[channel]).add_(RGB_MEAN[channel])
#     return image

def features_extract_network(name: str, *args, **kwargs) -> BaseArch:
    """返回特定类型的图片特征提取网络

    Args:
        name (str): 特征提取网络的名称，目前支持的网络名称有 vgg16, vgg19
        args: 图片特征提取网络对象创建的初始化参数选项
        kwargs: 图片特征提取网络对象创建的初始化参数选项
    """
    supported_name = ("vgg16", "vgg19")
    assert name in supported_name, f"未知网络模型名称 {name}，目前支持的有 {', '.join(supported_name)}"

    if name == "vgg16":
        return VGG16(*args, **kwargs)
    if name == "vgg19":
        return VGG19(*args, **kwargs)

# def data_parallel_network(network_arch: nn.Module, devices: list) -> nn.Module:
#     """数据并行的网络初始化
#
#     如若给定的 devices 中包含了 CPU 设备，那么直接返回原来的网络；若给定了多 GPU 设备，则对网络做单机多
#     GPU 并行处理。
#     """
#     return network_arch if devices is None or len(devices) == 0 or torch.device("cpu") in devices \
#         else nn.DataParallel(network_arch, device_ids=devices, output_device=devices[0])



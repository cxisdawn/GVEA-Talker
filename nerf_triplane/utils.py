import os
import glob
import tqdm
import math
import random
import warnings
import tensorboardX
from scipy.fft import dct

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import imageio
import lpips

# import fast_neural_style
from . import fast_neural_style
from pytorch_msssim import ms_ssim
from scipy.linalg import sqrtm
import scipy
from torchvision.models import inception_v3
from torchvision.transforms import functional as F2

from skimage.util import img_as_float
from skimage.metrics import normalized_root_mse
from skimage.transform import resize
from skimage.color import rgb2gray
# from skimage.feature import greycomatrix, greycoprops
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte

from skimage import img_as_float
from skimage.transform import resize
from skimage.color import rgb2gray
from brisque import BRISQUE


def blend_with_mask_cuda(src, dst, mask):
    src = src.permute(2, 0, 1)
    dst = dst.permute(2, 0, 1)
    mask = mask.unsqueeze(0)

    # Blending
    blended = src * mask + dst * (1 - mask)

    # Convert back to numpy and return
    return blended.permute(1, 2, 0).detach().cpu().numpy()


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    # 这个函数生成一个坐标网格，根据 PyTorch 的版本选择不同的调用方式。

    # 检查当前 PyTorch 版本
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        # 如果 PyTorch 版本低于 1.10，直接调用旧版的 torch.meshgrid
        return torch.meshgrid(*args)
    else:
        # 如果 PyTorch 版本为 1.10 或以上，使用新的参数 indexing 指定网格生成方式
        return torch.meshgrid(*args, indexing='ij')


def get_audio_features(features, att_mode, index):
    if att_mode == 0:
        return features[[index]]
    elif att_mode == 1:
        left = index - 8
        pad_left = 0
        if left < 0:
            pad_left = -left
            left = 0
        auds = features[left:index]
        if pad_left > 0:
            # pad may be longer than auds, so do not use zeros_like
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds],
                             dim=0)
        return auds
    elif att_mode == 2:
        left = index - 4
        right = index + 4
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = features[left:right]
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)  # [8, 16]
        return auds
    else:
        raise NotImplementedError(f'wrong att_mode: {att_mode}')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# copied from pytorch3d
def _angle_from_tan(
        axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str = 'XYZ') -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    # if len(convention) != 3:
    #     raise ValueError("Convention must have 3 letters.")
    # if convention[1] in (convention[0], convention[2]):
    #     raise ValueError(f"Invalid convention {convention}.")
    # for letter in convention:
    #     if letter not in ("X", "Y", "Z"):
    #         raise ValueError(f"Invalid letter {letter} in convention string.")
    # if matrix.size(-1) != 3 or matrix.size(-2) != 3:
    #     raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


@torch.cuda.amp.autocast(enabled=False)
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


@torch.cuda.amp.autocast(enabled=False)
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str = 'XYZ') -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    # print(euler_angles, euler_angles.dtype)

    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]

    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


@torch.cuda.amp.autocast(enabled=False)
def convert_poses(poses):
    # poses: [B, 4, 4]
    # return [B, 3], 4 rot, 3 trans
    out = torch.empty(poses.shape[0], 6, dtype=torch.float32, device=poses.device)
    out[:, :3] = matrix_to_euler_angles(poses[:, :3, :3])
    out[:, 3:] = poses[:, :3, 3]
    return out


@torch.cuda.amp.autocast(enabled=False)
def get_bg_coords(H, W, device):
    X = torch.arange(H, device=device) / (H - 1) * 2 - 1  # in [-1, 1]
    Y = torch.arange(W, device=device) / (W - 1) * 2 - 1  # in [-1, 1]
    xs, ys = custom_meshgrid(X, Y)
    bg_coords = torch.cat([xs.reshape(-1, 1), ys.reshape(-1, 1)], dim=-1).unsqueeze(0)  # [1, H*W, 2], in [-1, 1]
    return bg_coords


import numpy as np
from PIL import Image


def handle_mask(image):
    # 将图片转换为NumPy数组

    image = Image.open(image[0])
    # print("image=",image[0])
    image_array = np.array(image)

    # 检查数组数据类型并转换为数值类型（例如浮点数）以便进行比较
    if image_array.dtype.kind in {'U', 'S'}:  # 'U' 表示 Unicode 字符串, 'S' 表示字节字符串
        # 尝试将其转换为浮点数类型
        image_array = image_array.astype(np.float64)
    elif not np.issubdtype(image_array.dtype, np.number):
        # 如果仍然不是数值类型，则输出错误提示
        raise ValueError("输入的图像数组必须包含数值类型的像素数据。")

    # 创建非零像素掩码，判断哪些像素值大于0
    mask = image_array > 0

    # # 打印掩码的形状和内容
    # print(mask.shape)
    # print(mask)

    # 创建单通道掩码，判断在所有通道上是否有任何一个像素值非零
    single_channel_mask = np.any(mask, axis=-1)

    return single_channel_mask


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, torso_patch_size=16, train_torso=False, torso_img=None,
             rect=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''
    if torso_img is not None:
        torso_img = handle_mask(torso_img)  # 掩码变一个元素
    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    # 如果有框，则N是框内的像素数量
    if rect is not None:
        xmin, xmax, ymin, ymax = rect
        N = (xmax - xmin) * (ymax - ymin)
    # torch.linspace(0, W-1, W, device=device) 生成从 0 到 W-1 的 W 个均匀间隔的数值（包括 0 和 W-1）。
    # torch.linspace(0, H-1, H, device=device) 生成从 0 到 H-1 的 H 个均匀间隔的数值（包括 0 和 H-1）。
    # 这些数值表示图像平面的像素坐标。例如，对于宽度 W 和高度 H 的图像，
    # torch.linspace(0, W-1, W) 生成从 0 到 W-1 的列索引，
    # torch.linspace(0, H-1, H) 生成从 0 到 H-1 的行索引。
    # 如果 W = 4 和 H = 3，
    # i = tensor([[0., 1., 2., 3.],
    #             [0., 1., 2., 3.],
    #             [0., 1., 2., 3.]])
    # j = tensor([[0., 0., 0., 0.],
    #             [1., 1., 1., 1.],
    #             [2., 2., 2., 2.]])
    # i 和 j 的形状为 [H, W]。
    # i 的每一行都是从 0 到 W-1 的列索引。
    # j 的每一列都是从 0 到 H-1 的行索引。
    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device),
                           torch.linspace(0, H - 1, H, device=device))  # float
    # i.t()：对 i 进行转置操作，将其从形状 [H, W] 变为 [W, H]。
    # 将转置后的 i 展平成一个一维张量，并添加一个新的维度，变为 [1, H*W]。
    # expand([B, H*W])：将 [1, H*W] 的张量扩展为 [B, H*W]，
    # 其中 B 是批次大小。这样可以适应批处理操作，每个批次中的图像都有相同的像素网格坐标。
    # + 0.5：在像素坐标上加 0.5，将坐标从整数像素位置移动到每个像素的中心。这是为了在射线计算中提高精度，因为射线通常从像素中心发射。

    # i,j  [B, H*W]
    # i = tensor([[0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5, 3.5]]),
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)
        if train_torso is not True:
            if patch_size > 1:

                # random sample left-top cores.
                # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
                # num_patch 表示可以容纳的小块数量，它等于总的射线数量 N 除以每个小块的像素数量 patch_size ** 2。
                num_patch = N // (patch_size ** 2)
                inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
                inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
                # torch.stack([inds_x, inds_y], dim=-1) 将 inds_x 和 inds_y 堆叠成形状为 [num_patch, 2] 的张量，
                # 每个元素表示一个小块左上角的坐标。
                inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

                # create meshgrid for each patch
                # 表示每个小块中每个像素的相对坐标（相对于小块的左上角）。
                pi, pj = custom_meshgrid(torch.arange(patch_size, device=device),
                                         torch.arange(patch_size, device=device))
                offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]  # 每个元素表示小块中每个像素的相对坐标。
                # inds.unsqueeze(1) 将小块的左上角坐标扩展一个维度，变为 [num_patch, 1, 2]。
                # offsets.unsqueeze(0) 将每个小块中每个像素的相对坐标扩展一个维度，变为 [1, patch_size^2, 2]。
                # inds.unsqueeze(1) + offsets.unsqueeze(0) 计算每个小块中每个像素的实际坐标，
                # 结果是形状为 [num_patch, patch_size^2, 2] 的张量。
                inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
                inds = inds.view(-1, 2)  # [N, 2]  将张量展平成形状为 [N, 2] 的一维张量，其中 N = num_patch * patch_size^2。
                inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten 将二维坐标转换为一维索引。  index=i×W+j
                # # 结果
                # inds = tensor([ 8, 14, 20,  9, 15, 21, 10, 16, 22,
                #                22, 28, 34, 23, 29, 35, 24, 30, 36])
                inds = inds.expand([B, N])  # 使每个批次中的图像都有相同的像素索引。


            # only get rays in the specified rect
            elif rect is not None:
                # assert B == 1
                mask = torch.zeros(H, W, dtype=torch.bool, device=device)
                xmin, xmax, ymin, ymax = rect
                mask[xmin:xmax, ymin:ymax] = 1
                inds = torch.where(mask.view(-1))[0]  # [nzn]
                inds = inds.unsqueeze(0)  # [1, N]

            else:
                # 正常情况
                # 随机生成 N 个整数，范围从 0 到 H*W-1。
                # H*W 是图像中的总像素数。
                # 由于是随机生成，可能会有重复的索引。
                inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
                inds = inds.expand([B, N])
                # print("inds.shape=", inds.shape)  # inds.shape= torch.Size([1, 65536])
            i = torch.gather(i, -1, inds)  # [B,N]
            j = torch.gather(j, -1, inds)  # [B,N]



        else:  # 躯干

            # # 将图像张量展平成形状为 [B, H*W, C]
            # flat_img = torso_img.view(B, H*W, -1)

            # # 计算每个像素的通道和，并生成非零像素的掩码（布尔张量）
            # non_zero_mask = flat_img.sum(dim=-1) != 0

            # # 获取非零像素的索引，返回形状为 [num_non_zero_pixels, 2] 的张量
            # # 第一个维度为批次索引，第二个维度为扁平化后图像中的索引
            # non_zero_indices = non_zero_mask.nonzero(as_tuple=False)

            # # 计算非零像素在二维图像中的行坐标和列坐标
            # # 其中，non_zero_indices[:, 1] 是扁平化后图像的索引
            # non_zero_indices_2d = non_zero_indices[:, 1] // W, non_zero_indices[:, 1] % W

            # # 提取非零像素的批次索引
            # batch_indices = non_zero_indices[:, 0]

            # # 组合批次索引、行坐标和列坐标，形成非零像素的完整索引
            # non_zero_indices_combined = torch.stack([batch_indices, non_zero_indices_2d[0], non_zero_indices_2d[1]], dim=-1)

            # # 计算需要的补丁数量，每个补丁大小为 patch_size*patch_size
            # num_patch = N // (patch_size ** 2)

            # # 从非零像素的二维坐标中提取行坐标和列坐标，忽略批次索引
            # inds = non_zero_indices_combined[:, 1:]

            # # 随机从非零像素中选择补丁中心的位置
            # # 这里 torch.randint(0, inds.size(0), size=[num_patch], device=device) 用于随机选择 num_patch 个索引
            # sampled_inds = inds[torch.randint(0, inds.size(0), size=[num_patch], device=device)]

            # # 为每个补丁生成 patch_size 大小的网格，用于获取每个补丁的所有像素坐标
            # pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))

            # # 将生成的网格行列坐标堆叠起来，形成补丁内的偏移量 [p^2, 2]
            # offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)

            # # 将偏移量添加到采样的补丁中心位置，得到补丁内所有像素的坐标
            # sampled_inds = sampled_inds.unsqueeze(1) + offsets.unsqueeze(0)

            # # 将补丁内所有像素的坐标展平为 [N, 2] 形状，其中 N 为总像素数
            # sampled_inds = sampled_inds.view(-1, 2)

            # # 将 2D 坐标转换为扁平化后的索引
            # sampled_inds = sampled_inds[:, 0] * W + sampled_inds[:, 1]

            # # 将补丁索引扩展到批次大小
            # # 这里的 expand([B, N]) 将补丁索引扩展为 [B, N] 的形状，其中 B 是批次大小
            # sampled_inds = sampled_inds.expand([B, N])

            # # 打印最终的补丁索引
            # print(sampled_inds)

            # 假设你已经有了掩码图像 mask，形状为 [H, W]
            # 并且有一个形状为 [B, H, W] 的图像批次 torch.tensor([B, H, W])
            coords = np.argwhere(
                torso_img == 1)  # np.argwhere 返回一个数组，其中包含满足条件（这里是 image == 1）的所有坐标。返回的数组的形状为 (n, 2)，其中 n 是满足条件的元素的数量，每行包含 (y, x) 的坐标。
            # 计算过滤条件
            len_mask = len(coords)  # mask的像素点
            # print("len_mask",len_mask)
            condition = (0 <= coords[:, 0]) & (coords[:, 0] < H - torso_patch_size) & \
                        (0 <= coords[:, 1]) & (coords[:, 1] < W - torso_patch_size)

            # 过滤出符合条件的坐标对
            filtered_coords = coords[condition]
            len_mask = len(filtered_coords)  # mask的像素点
            # print("len_mask",len_mask)
            m = N / 2
            # print(m)
            num_patch = int(m // (torso_patch_size))
            # print("num_patch=",num_patch)
            # 从 0 到 len_mask之间随机生成 num_patch 个整数
            random_numbers = np.random.randint(0, len_mask, size=num_patch)
            inds1 = filtered_coords[random_numbers]  # [np,2]
            # 转换为 PyTorch 张量
            inds1 = torch.from_numpy(inds1).to(device)  # [np, 2]
            # print("type of inds",type(inds1))
            pi, pj = custom_meshgrid(torch.arange(torso_patch_size, device=device),
                                     torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]
            # print("inds1",inds1.shape)
            inds1 = inds1.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            # print("inds1",inds1.shape)
            inds1 = inds1.view(-1, 2)  # [N, 2]
            inds1 = inds1[:, 0] * W + inds1[:, 1]  # [N], flatten
            # print("inds1",inds1.shape)

            num_patch = int(m // (patch_size ** 2))
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds2 = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds2 = inds2.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds2 = inds2.view(-1, 2)  # [N, 2]
            inds2 = inds2[:, 0] * W + inds2[:, 1]  # [N], flatten

            # 合并 inds 和 inds2
            inds_combined = torch.cat((inds1, inds2), dim=0)  # [N], 合并后大小为 N

            # 扩展合并后的 inds
            inds_combined = inds_combined.expand([B, inds_combined.size(0)])
            inds = inds_combined
            # print("inds.shape=",inds.shape)
            i = torch.gather(i, -1, inds)  # [B,N]
            j = torch.gather(j, -1, inds)  # [B,N]



    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])
    # 代码根据索引 inds 从坐标网格 i 和 j 中选择特定的坐标。这使得可以获取到指定像素位置的坐标，
    results['i'] = i
    results['j'] = j
    results['inds'] = inds
    # zs 是一个与 i 形状相同的张量，所有元素都设置为 1。
    zs = torch.ones_like(i)  # [B,N]  这里 zs 表示深度值（通常在相机坐标系中），在这里我们假设深度为 1。实际应用中，可以根据需要调整 zs。
    # 计算相机坐标系中的 x 和 y 坐标
    # i 和 j 是图像中的像素坐标。
    # cx 和 cy 是相机的主点坐标（即相机坐标系的原点在图像平面上的位置），通常是图像宽度和高度的中心。
    # fx 和 fy 是相机的焦距（焦距在 x 和 y 方向上的分量).
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    # 这样可以得到每个像素的方向向量，方便在渲染或光线追踪等任务中使用。
    directions = torch.stack((xs, ys, zs), dim=-1)  # 示每个像素的光线方向  [B, N, 3]，其中 3 表示 x, y 和 z 方向的分量。
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)  # 标准化光线方向,使得每个方向向量的长度为 1。

    # 得到的 rays_d 表示在世界坐标系下的光线方向向量。
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    # poses[..., :3, 3] 提取 poses 张量中的最后一列，表示相机的位置（即相机在世界坐标系中的原点），形状为 [B, 3]。
    rays_o = poses[..., :3, 3]  # [B, 3]
    # rays_o[..., None, :] 在 rays_o 的最后一个维度上添加一个新的维度，形状变为 [B, 1, 3]。
    #  将 rays_o 张量扩展到与 rays_d 张量相同的形状 [B, N, 3]。这使得每个光线的起点在批次中的所有像素上都是一样的。
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys),
                                                  len(zs)).detach().cpu().numpy()  # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


from pytorch_msssim import ms_ssim



class MSSSIMMeter:
    def __init__(self, data_range=1.0, size_average=True):
        self.V = 0
        self.N = 0
        self.data_range = data_range
        self.size_average = size_average

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, H, W, 3] --> [B, 3, H, W]
            inp = inp.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W]
        msssim_value = ms_ssim(preds, truths, data_range=self.data_range, size_average=self.size_average).item()
        self.V += msssim_value
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "MS-SSIM"), self.measure(), global_step)

    def report(self):
        return f'MS-SSIM = {self.measure():.6f}'


# ---------------
class NIQEMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, H, W, 3] --> [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def compute_niqe(self, img):
        img = img_as_float(img)
        img = rgb2gray(img)
        img = resize(img, (256, 256), anti_aliasing=True)
        # 将浮点图像转换为 uint8 类型
        img_uint8 = img_as_ubyte(img)

        # 然后再使用 greycomatrix 函数
        glcm = graycomatrix(img_uint8, [1], [0], 256, symmetric=True, normed=True)

        # glcm = greycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]
        niqe_score = (contrast + dissimilarity + homogeneity + energy + correlation + asm) / 6
        return niqe_score

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W]
        preds = preds.permute(0, 2, 3, 1).cpu().numpy()  # [B, 3, H, W] --> [B, H, W, 3]
        truths = truths.permute(0, 2, 3, 1).cpu().numpy()  # [B, 3, H, W] --> [B, H, W, 3]

        for pred, truth in zip(preds, truths):
            niqe_value = self.compute_niqe(pred)
            self.V += niqe_value
            self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "NIQE"), self.measure(), global_step)

    def report(self):
        return f'NIQE = {self.measure():.6f}'


# =-====================================
# class FIDMeter:
#     def __init__(self, device=None):
#         self.V = 0
#         self.N = 0
#         self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.inception = inception_v3(pretrained=True, transform_input=False).to(self.device).eval()
#         self.reset()
#
#     def reset(self):
#         self.real_features = []
#         self.fake_features = []
#
#     def clear(self):
#         self.V = 0
#         self.N = 0
#         self.reset()
#
#     def prepare_inputs(self, *inputs):
#         outputs = []
#         for i, inp in enumerate(inputs):
#             inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, H, W, 3] --> [B, 3, H, W]
#             inp = F2.resize(inp, (299, 299))  # Resize to InceptionV3 input size
#             inp = inp.to(self.device)
#             outputs.append(inp)
#         return outputs
#
#     def get_features(self, x):
#         with torch.no_grad():
#             x = self.inception(x)
#         return x
#
#     def update(self, preds, truths):
#         preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W]
#         fake_features = self.get_features(preds)
#         real_features = self.get_features(truths)
#         self.fake_features.append(fake_features.cpu().numpy())
#         self.real_features.append(real_features.cpu().numpy())
#
#     def compute_statistics(self, features):
#         features = np.concatenate(features, axis=0)
#         mu = np.mean(features, axis=0)
#         sigma = np.cov(features, rowvar=False)
#         return mu, sigma
#
#     def measure(self):
#         mu1, sigma1 = self.compute_statistics(self.real_features)
#         mu2, sigma2 = self.compute_statistics(self.fake_features)
#         ssdiff = np.sum((mu1 - mu2) ** 2.0)
#         covmean = sqrtm(sigma1.dot(sigma2))
#         if np.iscomplexobj(covmean):
#             covmean = covmean.real
#         fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
#         return fid
#
#     def write(self, writer, global_step, prefix=""):
#         writer.add_scalar(os.path.join(prefix, "FID"), self.measure(), global_step)
#
#     def report(self):
#         return f'FID = {self.measure():.6f}'
class FIDMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inception = inception_v3(pretrained=True, transform_input=False).to(self.device).eval()
        self.reset()

    def reset(self):
        self.real_features = []
        self.fake_features = []

    def clear(self):
        self.V = 0
        self.N = 0
        self.reset()

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, H, W, 3] --> [B, 3, H, W]
            inp = F2.resize(inp, (299, 299))  # Resize to InceptionV3 input size
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def get_features(self, x):
        with torch.no_grad():
            x = self.inception(x)
        return x

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W]
        fake_features = self.get_features(preds)
        real_features = self.get_features(truths)
        self.fake_features.append(fake_features.cpu().numpy())
        self.real_features.append(real_features.cpu().numpy())

    def compute_statistics(self, features):
        features = np.concatenate(features, axis=0)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def measure(self):
        mu1, sigma1 = self.compute_statistics(self.real_features)
        mu2, sigma2 = self.compute_statistics(self.fake_features)
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "FID"), self.measure(), global_step)

    def report(self):
        return f'FID = {self.measure():.6f}'



# -----------------------


class BRISQUEMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.brisque = BRISQUE()

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, H, W, 3] --> [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def compute_brisque(self, img):
        img = img_as_float(img)

        # 检查输入数组的形状
        if img.ndim == 2:
            # 如果输入数组是灰度图像，扩展维度以匹配期望的形状
            img = np.stack((img, img, img), axis=-1)
        elif img.shape[-1] != 3:
            raise ValueError("输入数组必须具有3个颜色通道")

        img = resize(img, (256, 256), anti_aliasing=True)
        brisque_score = self.brisque.score(img)
        return brisque_score

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W]
        preds = preds.permute(0, 2, 3, 1).cpu().numpy()  # [B, 3, H, W] --> [B, H, W, 3]
        truths = truths.permute(0, 2, 3, 1).cpu().numpy()  # [B, 3, H, W] --> [B, H, W, 3]

        for pred, truth in zip(preds, truths):
            brisque_value = self.compute_brisque(pred)
            self.V += brisque_value
            self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "BRISQUE"), self.measure(), global_step)

    def report(self):
        return f'BRISQUE = {self.measure():.6f}'


# --------------------------------------------===============
class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        # print('preds',preds.shape)
        # print(preds)
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range in [0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item()  # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'


class LMDMeter:
    def __init__(self, backend='dlib', region='mouth'):
        self.backend = backend
        self.region = region  # mouth or face

        if self.backend == 'dlib':
            import dlib

            # load checkpoint manually
            self.predictor_path = './shape_predictor_68_face_landmarks.dat'
            if not os.path.exists(self.predictor_path):
                raise FileNotFoundError(
                    'Please download dlib checkpoint from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')

            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.predictor_path)

        else:

            import face_alignment
            try:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            except:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        self.V = 0
        self.N = 0

    def get_landmarks(self, img):

        if self.backend == 'dlib':
            dets = self.detector(img, 1)
            for det in dets:
                shape = self.predictor(img, det)
                # ref: https://github.com/PyImageSearch/imutils/blob/c12f15391fcc945d0d644b85194b8c044a392e0a/imutils/face_utils/helpers.py
                lms = np.zeros((68, 2), dtype=np.int32)
                for i in range(0, 68):
                    lms[i, 0] = shape.part(i).x
                    lms[i, 1] = shape.part(i).y
                break

        else:
            lms = self.predictor.get_landmarks(img)[-1]

        # self.vis_landmarks(img, lms)
        lms = lms.astype(np.float32)

        return lms

    def vis_landmarks(self, img, lms):
        plt.imshow(img)
        plt.plot(lms[48:68, 0], lms[48:68, 1], marker='o', markersize=1, linestyle='-', lw=2)
        plt.show()

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.detach().cpu().numpy()
            inp = (inp * 255).astype(np.uint8)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        # assert B == 1
        preds, truths = self.prepare_inputs(preds[0], truths[0])  # [H, W, 3] numpy array

        # get lms
        lms_pred = self.get_landmarks(preds)
        lms_truth = self.get_landmarks(truths)

        if self.region == 'mouth':
            lms_pred = lms_pred[48:68]
            lms_truth = lms_truth[48:68]

        # avarage
        lms_pred = lms_pred - lms_pred.mean(0)
        lms_truth = lms_truth - lms_truth.mean(0)

        # distance
        dist = np.sqrt(((lms_pred - lms_truth) ** 2).sum(1)).mean(0)

        self.V += dist
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LMD ({self.backend})"), self.measure(), global_step)

    def report(self):
        return f'LMD ({self.backend}) = {self.measure():.6f}'


# --------------------------------------------------------------===========
import torch
import numpy as np
import os
import face_alignment


class LSEC:
    def __init__(self, backend='face_alignment', device=None):
        self.backend = backend
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.V = 0
        self.N = 0

        if self.backend == 'face_alignment':
            try:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            except:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for inp in inputs:
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def get_landmarks(self, img):
        img = img.permute(1, 2, 0).cpu().numpy()  # [3, H, W] -> [H, W, 3]
        img = (img * 255).astype(np.uint8)  # 将图像转换为uint8类型
        lms = self.predictor.get_landmarks(img)[-1]
        return lms

    def calculate_lsec(self, preds, truths):
        preds, truths = preds.cpu().numpy(), truths.cpu().numpy()
        lsec_values = []
        for pred, truth in zip(preds, truths):
            lms_pred = self.get_landmarks(torch.tensor(pred))
            lms_truth = self.get_landmarks(torch.tensor(truth))
            lms_pred = lms_pred[48:68]  # 只取嘴部关键点
            lms_truth = lms_truth[48:68]
            dist = np.abs(lms_pred - lms_truth).mean(0)  # 计算平均绝对误差
            lsec_values.append(dist)
        return np.mean(lsec_values)

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.calculate_lsec(preds, truths)
        self.V += v
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "LSE-C"), self.measure(), global_step)

    def report(self):
        return f'LSE-C = {self.measure():.6f}'

class LSED:
    def __init__(self, backend='face_alignment', device=None):
        self.backend = backend
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.V = 0  # 用于累积误差值
        self.N = 0  # 用于统计总的样本数

        # 初始化人脸关键点预测器
        if self.backend == 'face_alignment':
            try:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            except:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    def clear(self):
        """清除累积的误差值和样本数"""
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        """将输入数据转化为 [B, 3, H, W] 格式并移动到指定设备"""
        outputs = []
        for inp in inputs:
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, H, W, 3] -> [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def get_landmarks(self, img):
        """从图像中提取面部关键点"""
        img = img.permute(1, 2, 0).cpu().numpy()  # [3, H, W] -> [H, W, 3]
        img = (img * 255).astype(np.uint8)  # 将图像转换为 uint8 类型
        lms = self.predictor.get_landmarks(img)[-1]
        return lms

    def calculate_lsed(self, preds, truths):
        """计算 LSED 误差"""
        preds, truths = preds.cpu().numpy(), truths.cpu().numpy()
        lsed_values = []
        for pred, truth in zip(preds, truths):
            lms_pred = self.get_landmarks(torch.tensor(pred))
            lms_truth = self.get_landmarks(torch.tensor(truth))
            lms_pred = lms_pred[48:68]  # 只取嘴部关键点
            lms_truth = lms_truth[48:68]

            # 计算预测和真实关键点的两两欧氏距离
            dist_pred = np.linalg.norm(lms_pred[:, None] - lms_pred[None, :], axis=-1)
            dist_truth = np.linalg.norm(lms_truth[:, None] - lms_truth[None, :], axis=-1)

            # 计算距离矩阵的平均绝对误差
            dist_error = np.abs(dist_pred - dist_truth).mean()
            lsed_values.append(dist_error)

        return np.mean(lsed_values)

    def update(self, preds, truths):
        """更新累计的误差值"""
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W]
        v = self.calculate_lsed(preds, truths)
        self.V += v
        self.N += 1

    def measure(self):
        """计算当前的 LSED 平均值"""
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        """将 LSED 写入到日志中"""
        writer.add_scalar(os.path.join(prefix, "LSE-D"), self.measure(), global_step)

    def report(self):
        """返回当前 LSED 的报告字符串"""
        return f'LSE-D = {self.measure():.6f}'


# --------------------------------------------------------------===========
# class AUEMeter:
#     def __init__(self, device=None):
#         self.V = 0
#         self.N = 0
#         self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     def clear(self):
#         self.V = 0
#         self.N = 0

#     def prepare_inputs(self, *inputs):
#         outputs = []
#         for inp in inputs:
#             inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, H, W, 3] --> [B, 3, H, W]
#             inp = inp.to(self.device)
#             outputs.append(inp)
#         return outputs

#     def compute_aue(self, pred, truth):
#         aue_value = np.mean(np.abs(pred - truth))
#         return aue_value

#     def update(self, preds, truths):
#         preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W]
#         preds = preds.permute(0, 2, 3, 1).cpu().numpy()  # [B, 3, H, W] --> [B, H, W, 3]
#         truths = truths.permute(0, 2, 3, 1).cpu().numpy()  # [B, 3, H, W] --> [B, H, W, 3]

#         for pred, truth in zip(preds, truths):
#             aue_value = self.compute_aue(pred, truth)
#             self.V += aue_value
#             self.N += 1

#     def measure(self):
#         return self.V / self.N

#     def write(self, writer, global_step, prefix=""):
#         writer.add_scalar(os.path.join(prefix, "AUE"), self.measure(), global_step)

#     def report(self):
#         return f'AUE = {self.measure():.6f}'


import face_alignment


class AUEMeter:
    def __init__(self, backend='face_alignment', device=None):
        self.backend = backend
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.V = 0
        self.N = 0

        if self.backend == 'face_alignment':
            try:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            except:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for inp in inputs:
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def get_landmarks(self, img):
        img = img.permute(1, 2, 0).cpu().numpy()  # [3, H, W] -> [H, W, 3]
        img = (img * 255).astype(np.uint8)  # 将图像转换为uint8类型
        lms = self.predictor.get_landmarks(img)[-1]
        return lms

    def calculate_aue(self, preds, truths):
        preds, truths = preds.cpu().numpy(), truths.cpu().numpy()
        aue_values = []
        for pred, truth in zip(preds, truths):
            lms_pred = self.get_landmarks(torch.tensor(pred))
            lms_truth = self.get_landmarks(torch.tensor(truth))
            lms_pred = lms_pred[48:68]  # 只取嘴部关键点
            lms_truth = lms_truth[48:68]
            dist = np.sqrt(((lms_pred - lms_truth) ** 2).sum(1)).mean(0)
            aue_values.append(dist)
        return np.mean(aue_values)

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.calculate_aue(preds, truths)
        self.V += v
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "AUE"), self.measure(), global_step)

    def report(self):
        return f'AUE = {self.measure():.6f}'


# --------------------------------------------------------------===========

class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network

                 perceptual_loss_func,  # 感知损失
                 tv_loss,

                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 ema_update_interval=1000,  # update ema per $ training steps.
                 lr_scheduler=None,  # scheduler
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 ):

        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.ema_update_interval = ema_update_interval
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.flip_finetune_lips = self.opt.finetune_lips
        self.flip_init_lips = self.opt.init_lips
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        self.count = 0
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1 or self.opt.finetune_lips or True:
            import lpips
            # self.criterion_lpips_vgg = lpips.LPIPS(net='vgg').to(self.device)
            self.criterion_lpips_alex = lpips.LPIPS(net='alex').to(self.device)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        # TODO
        """
          (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
        """
        if self.opt.use_percep_loss:
            self.features_net = fast_neural_style.utils.features_extract_network(self.opt.arch, self.opt.content_layers,
                                                                                 self.opt.style_layers).to(self.device)
            self.content_loss_func, self.style_loss_func = perceptual_loss_func
            self.tv_loss = tv_loss
        self.emo_lstm_h0 = None
        self.emo_lstm_c0 = None

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data):
        # B就是每次取的帧数
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        bg_coords = data['bg_coords']  # [1, N, 2]
        poses = data['poses']  # [B, 6]
        face_mask = data['face_mask']  # [B, N]
        eye_mask = data['eye_mask']  # [B, N]
        lhalf_mask = data['lhalf_mask']
        eye = data['eye']  # [B, 1]
        auds = data['auds']  # [B, 29, 16] now [B,1,512]

        if self.opt.contrast_loss:
            audneg = data['rev_aud']  # now[8,1,512]
            audact = data['act_aud']  # [B, 29, 16]  now[8,1,512]
            emo_feat = data['emo_feat']  # [B,1024] now [1,1024]
        index = data['index']  # [B]  第几帧
        # print('audact.shape',audact.shape)  #  [8,1,512]
        # print('audneg',audneg.shape)  #  [8,1,512]
        # TODO ADDING

        if self.opt.emo == True:
            # emo_label_old = data['emo_label']  # [B,1]
            emo_label = data['emo_label'] # [B,9]
            # print('emo_label',emo_label.shape)
        else:
            emo_label = None
        if not self.opt.torso:
            rgb = data['images']  # [B, N, 3]
        else:
            rgb = data['bg_torso_color']

        B, N, C = rgb.shape

        if self.opt.color_space == 'linear':
            rgb[..., :3] = srgb_to_linear(rgb[..., :3])

        bg_color = data['bg_color']
        if not self.opt.torso:
            outputs = self.model.render(rays_o, rays_d, auds, bg_coords, poses, emo_label=emo_label, eye=eye,
                                        index=index, staged=False, bg_color=bg_color, perturb=True,
                                        force_all_rays=False if (
                                                self.opt.patch_size <= 1 and not self.opt.train_camera) else True,
                                        **vars(self.opt))
        else:
            outputs = self.model.render_torso(rays_o, rays_d, auds, bg_coords, poses, eye=eye, index=index,
                                              staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if (
                        self.opt.patch_size <= 1 and not self.opt.train_camera) else True, **vars(self.opt))

        if not self.opt.torso:
            pred_rgb = outputs['image']
        else:
            pred_rgb = outputs['torso_color']

        # loss factor
        step_factor = min(self.global_step / self.opt.iters, 1.0)

        # MSE loss
        loss = self.criterion(pred_rgb, rgb).mean(-1)  # [B, N, 3] --> [B, N]

        # TODO
        if self.opt.use_percep_loss:
            # 将图像的颜色空间从线性空间变成srgb空间
            pred = pred_rgb.clone()
            ori_rgb = rgb.clone()
            # 采样光线数进行开方
            # print(pred.shape)
            H = int(math.sqrt(pred.shape[1]))
            W = int(math.sqrt(pred.shape[1]))
            # 对输出进行reshape成图像的形状（H,W,3）,然后变成(3,H,W)
            pred_rgb_reshape = pred[0].reshape(H, W, 3).permute(2, 0, 1)
            rgb_reshape = ori_rgb[0].reshape(H, W, 3).permute(2, 0, 1)
            # 求content loss
            content_pred, style_pred = self.features_net(pred_rgb_reshape)
            content_ori, style_ori = self.features_net(rgb_reshape)
            content_loss = self.content_loss_func(content_ori, content_pred)

            # 求tv_loss
            # loss_2 = self.tv_loss(pred_rgb_reshape.unsqueeze(0))
            # 进行反向传播
            loss_percep = self.opt.content_weight * content_loss
        else:
            loss_percep = np.array([0])

        if self.opt.torso:

            loss = loss.mean()
            if self.opt.use_percep_loss:
                loss = loss + 0.1 * loss_percep
            loss += ((1 - self.model.anchor_points[:, 3]) ** 2).mean()
            return pred_rgb, rgb,  {'total_loss': loss, 'perceptual_loss': loss_percep,
                               }

        # camera optim regularization
        # if self.opt.train_camera:
        #     cam_reg = self.model.camera_dR[index].abs().mean() + self.model.camera_dT[index].abs().mean()
        #     loss = loss + 1e-2 * cam_reg

        if self.opt.unc_loss and not self.flip_finetune_lips:
            alpha = 0.2
            uncertainty = outputs['uncertainty']  # [N], abs sum
            beta = uncertainty + 1

            unc_weight = F.softmax(uncertainty, dim=-1) * N
            # print(unc_weight.shape, unc_weight.max(), unc_weight.min())
            loss *= alpha + (1 - alpha) * ((1 - step_factor) + step_factor * unc_weight.detach()).clamp(0, 10)
            # loss *= unc_weight.detach()

            beta = uncertainty + 1
            norm_rgb = torch.norm((pred_rgb - rgb), dim=-1).detach()
            loss_u = norm_rgb / (2 * beta ** 2) + (torch.log(beta) ** 2) / 2
            loss_u *= face_mask.view(-1)
            loss += step_factor * loss_u

            loss_static_uncertainty = (uncertainty * (~face_mask.view(-1)))
            loss += 1e-3 * step_factor * loss_static_uncertainty
        else:
            loss_u = np.array([0])
            loss_static_uncertainty = np.array([0])
        loss_lpips = 0
        # patch-based rendering
        if self.opt.patch_size > 1 and not self.opt.finetune_lips:
            rgb = rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous() * 2 - 1
            pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1,
                                                                                              2).contiguous() * 2 - 1

            # torch_vis_2d(rgb[0])
            # torch_vis_2d(pred_rgb[0])

            # LPIPS loss ?
            loss_lpips = self.criterion_lpips_alex(pred_rgb, rgb)
            loss = loss + 0.1 * loss_lpips
            loss_lpips = loss_lpips.mean()
        # lips finetune
        if self.opt.finetune_lips:
            xmin, xmax, ymin, ymax = data['rect']
            rgb = rgb.view(-1, xmax - xmin, ymax - ymin, 3).permute(0, 3, 1, 2).contiguous() * 2 - 1
            pred_rgb = pred_rgb.view(-1, xmax - xmin, ymax - ymin, 3).permute(0, 3, 1, 2).contiguous() * 2 - 1

            padding_h = max(0, (32 - rgb.shape[-2] + 1) // 2)
            padding_w = max(0, (32 - rgb.shape[-1] + 1) // 2)

            if padding_w or padding_h:
                rgb = torch.nn.functional.pad(rgb, (padding_w, padding_w, padding_h, padding_h))
                pred_rgb = torch.nn.functional.pad(pred_rgb, (padding_w, padding_w, padding_h, padding_h))

            # torch_vis_2d(rgb[0])
            # torch_vis_2d(pred_rgb[0])
            loss_lpips = self.criterion_lpips_alex(pred_rgb, rgb)
            # LPIPS loss
            loss = loss + 0.01 * loss_lpips
            loss_lpips = loss_lpips.mean()
        # flip every step... if finetune lips
        if self.flip_finetune_lips:
            self.opt.finetune_lips = not self.opt.finetune_lips

        loss = loss.mean()

        # if self.opt.use_percep_loss and self. opt.finetune_lips:
        # loss = loss + loss_percep
        if self.opt.use_percep_loss and not self.opt.finetune_lips:
            loss = loss + loss_percep
            # print(1111, self.opt.finetune_lips)

        # weights_sum loss
        # entropy to encourage weights_sum to be 0 or 1.
        if self.opt.torso:
            alphas = outputs['torso_alpha'].clamp(1e-5, 1 - 1e-5)
            # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
            loss_ws = - alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)
            loss = loss + 1e-4 * loss_ws.mean()

        else:
            alphas = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
            loss_ws = - alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)
            loss = loss + 1e-4 * loss_ws.mean()

        # aud att loss (regions out of face should be static)
        if self.opt.amb_aud_loss and not self.opt.torso:
            ambient_aud = outputs['ambient_aud']
            loss_amb_aud = (ambient_aud * (~face_mask.view(-1))).mean()
            # gradually increase it
            lambda_amb = step_factor * self.opt.lambda_amb
            loss += lambda_amb * loss_amb_aud
        else:
            loss_amb_aud = 0
        # eye att loss
        if self.opt.amb_eye_loss and not self.opt.torso:
            ambient_eye = outputs['ambient_eye'] / self.opt.max_steps

            loss_cross = ((ambient_eye * ambient_aud.detach()) * face_mask.view(-1)).mean()
            loss += lambda_amb * loss_cross
        else:
            loss_cross = 0
        # TODO emo att loss
        # emo att loss
        contrast_loss = np.array([0])
        if self.opt.emo and not self.opt.torso and not self.opt.finetune_lips:
            ambient_emo = outputs['ambient_emo'] / self.opt.max_steps
            emoout = (ambient_emo * (~face_mask.view(-1))).mean()

            emo2 = (((ambient_emo * ambient_aud.detach()) * (face_mask.view(-1)))).mean()
            # loss_cross_emo = emo2 * 0.0001 + emoout * 0.9999
            if -1 <= emoout <= 1 and -1<= emo2 <= 1:

                loss_cross_emo = emo2 * 0.05 + emoout * 0.999
            else:
                loss_cross_emo = F.sigmoid(emo2) * 0.001 + F.sigmoid(emoout) * 0.999
            if self.count // 3000 == 1:
                print(1111111111111111111111111111111111111111111111111111111)
                print('lambda_amb', lambda_amb)
                print('loss:', loss)
                print('eye_loss_cross', loss_cross)
                print('emo_loss_cross', loss_cross_emo)
                print(2222222222222222222222222222222222222222222222222222222)
                self.count = 0
            self.count += 1
            loss += lambda_amb * loss_cross_emo
        else:
            loss_cross_emo = np.array([0])
        # 实例化contrast网络
        if self.opt.contrast_loss and not self.opt.torso and not self.opt.finetune_lips:
                # print(audact)
                # print(audneg)
                # print(emo_label)
                contrast_loss = self.model.contrast(audact, audneg, emo_feat)
                loss += lambda_amb * contrast_loss * 0.05
                # if self.count // 3001 == 1:
                #     print(contrast_loss)
                #     self.count = 0

                # print('use contrast_loss')


                # print(contrast_loss)
        # TODO 规范化损失，需要将emo等新加进去的也写在这里
        if self.global_step % 16 == 0 and not self.flip_finetune_lips:
            xyzs, dirs, enc_a, ind_code, eye = outputs['rays']
            emo_label = outputs['emo_label']
            xyz_delta = (torch.rand(size=xyzs.shape, dtype=xyzs.dtype, device=xyzs.device) * 2 - 1) * 1e-3
            with torch.no_grad():

                sigmas_raw, rgbs_raw, ambient_aud_raw, ambient_eye_raw, ambient_emo_raw, unc_raw = self.model(xyzs,
                                                                                                              dirs,
                                                                                                              enc_a.detach(),
                                                                                                              ind_code.detach(),
                                                                                                              eye,
                                                                                                              emo_label=emo_label)

            sigmas_reg, rgbs_reg, ambient_aud_reg, ambient_eye_reg, ambient_emo_reg, unc_reg = self.model(
                xyzs + xyz_delta, dirs,
                enc_a.detach(),
                ind_code.detach(), eye,
                emo_label=emo_label)

            lambda_reg = step_factor * 1e-5
            reg_loss = 0
            if self.opt.unc_loss:
                reg_loss += self.criterion(unc_raw, unc_reg).mean()
            if self.opt.amb_aud_loss:
                reg_loss += self.criterion(ambient_aud_raw, ambient_aud_reg).mean()
            if self.opt.amb_eye_loss:
                reg_loss += self.criterion(ambient_eye_raw, ambient_eye_reg).mean()

            if not self.opt.finetune_lips:
                if self.opt.emo:
                    # print('使用规范化损失')
                    reg_loss += self.criterion(ambient_emo_raw, ambient_emo_reg).mean()
            else:

                pass
                # if self.opt.amb_emo_loss:
                #     reg_loss += self.criterion(ambient_emo_raw, ambient_emo_reg).mean()

            loss += reg_loss * lambda_reg
        else:
            reg_loss = 0

        return pred_rgb, rgb,  {'total_loss': loss, 'perceptual_loss': loss_percep,
                               'uncertainty_loss': loss_u.mean(), 'lpips_loss': loss_lpips,'eye_loss':loss_cross,'loss_amb_aud':loss_amb_aud,
                               'reg_loss':reg_loss,"emo_loss":loss_cross_emo,'contrast_loss':contrast_loss,"weight":loss_ws.mean(),'uncertainty':loss_static_uncertainty.mean()}

    def eval_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        bg_coords = data['bg_coords']  # [1, N, 2]
        poses = data['poses']  # [B, 7]
        images = data['images']  # [B, H, W, 3/4]
        if self.opt.portrait:
            images = data['bg_gt_images']
        auds = data['auds']
        index = data['index']  # [B]
        eye = data['eye']  # [B, 1]
        if self.opt.emo:
            emo_label = data['emo_label']  # [B,1]
        else:
            emo_label = None
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        # bg_color = 1
        bg_color = data['bg_color']

        outputs = self.model.render(rays_o, rays_d, auds, bg_coords, poses, eye=eye, emo_label=emo_label, index=index,
                                    staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_ambient_aud = outputs['ambient_aud'].reshape(B, H, W)
        pred_ambient_eye = outputs['ambient_eye'].reshape(B, H, W)

        pred_ambient_emo = outputs['ambient_emo'].reshape(B, H, W)

        pred_uncertainty = outputs['uncertainty'].reshape(B, H, W)

        loss_raw = self.criterion(pred_rgb, images)
        loss = loss_raw.mean()

        return pred_rgb, pred_depth, pred_ambient_aud, pred_ambient_eye, pred_ambient_emo, pred_uncertainty, images, loss, loss_raw

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        bg_coords = data['bg_coords']  # [1, N, 2]
        poses = data['poses']  # [B, 7]

        auds = data['auds']  # [B, 29, 16]
        index = data['index']
        H, W = data['H'], data['W']
        
        if self.opt.emo == True:
            emo_label = data['emo_label']  # [B,1]
        else:
            emo_label = None

        # allow using a fixed eye area (avoid eye blink) at test
        if self.opt.exp_eye and self.opt.fix_eye >= 0:
            eye = torch.FloatTensor([self.opt.fix_eye]).view(1, 1).to(self.device)
        else:
            eye = data['eye']  # [B, 1]

        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        else:
            bg_color = data['bg_color']

        self.model.testing = True
        outputs = self.model.render(rays_o, rays_d, auds, bg_coords, poses, emo_label=emo_label, eye=eye, index=index,
                                    staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))
        self.model.testing = False

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:],
                                               resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False)  # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_image=False):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_preds = []
        all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                path = os.path.join(save_path, f'{name}_{i:04d}_rgb.png')
                path_depth = os.path.join(save_path, f'{name}_{i:04d}_depth.png')

                # self.log(f"[INFO] saving test image to {path}")

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                if self.opt.portrait:
                    pred = blend_with_mask_cuda(preds[0], data["bg_gt_images"].squeeze(0),
                                                data["bg_face_mask"].squeeze(0))
                    pred = (pred * 255).astype(np.uint8)
                else:
                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_image:
                    imageio.imwrite(path, pred)
                    imageio.imwrite(path_depth, pred_depth)

                all_preds.append(pred)
                all_preds_depth.append(pred_depth)

                pbar.update(loader.batch_size)

        # write video
        all_preds = np.stack(all_preds, axis=0)
        all_preds_depth = np.stack(all_preds_depth, axis=0)
        imageio.mimwrite(os.path.join(save_path, f'{name}.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8,
                         macro_block_size=1)

        if self.opt.video_aud != '':
            os.system(
                f'ffmpeg -i {os.path.join(save_path, f"{name}.mp4")} -i {self.opt.video_aud} -strict -2 -c:v copy {os.path.join(save_path, f"{name}_audio.mp4")} -y')

        self.log(f"==> Finished Test.")

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

            if self.ema is not None and self.global_step % self.ema_update_interval == 0:
                self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, auds, eye=None, index=0, bg_color=None, spp=1, downscale=1):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        if auds is not None:
            auds = auds.to(self.device)

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        rays = get_rays(pose, intrinsics, rH, rW, -1)

        bg_coords = get_bg_coords(rH, rW, self.device)

        if eye is not None:
            eye = torch.FloatTensor([eye]).view(1, 1).to(self.device)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
            'auds': auds,
            'index': [index],  # support choosing index for individual codes
            'eye': eye,
            'poses': pose,
            'bg_coords': bg_coords,
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                # face: do not perturb for the first spp, else lead to scatters.
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='bilinear').permute(0, 2, 3,
                                                                                                   1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    # [GUI] test with provided data
    def test_gui_with_data(self, data, W, H):

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                # face: do not perturb for the first spp, else lead to scatters.
                preds, preds_depth = self.test_step(data, perturb=False)

        if self.ema is not None:
            self.ema.restore()

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        # the H/W in data may be differnt to GUI, so we still need to resize...
        preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='bilinear').permute(0, 2, 3, 1).contiguous()
        preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    # def train_one_epoch(self, loader):
    #     self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")
    #
    #     total_loss = 0
    #     if self.local_rank == 0 and self.report_metric_at_train:
    #         for metric in self.metrics:
    #             metric.clear()
    #
    #     self.model.train()
    #
    #     # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
    #     # ref: https://pytorch.org/docs/stable/data.html
    #     if self.world_size > 1:
    #         loader.sampler.set_epoch(self.epoch)
    #
    #     if self.local_rank == 0:
    #         pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, mininterval=1,
    #                          bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    #
    #     self.local_step = 0
    #     # 每一轮的开始 重新初始化隐藏状态
    #     # TODO emo的lstm状态初始化
    #     if self.opt.emo_lstm:
    #         self.model.emo_lstm_h0 = 0
    #         self.model.emo_lstm_c0 = 0
    #     for data in loader:
    #         # update grid every 16 steps
    #         if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
    #             with torch.cuda.amp.autocast(enabled=self.fp16):
    #                 self.model.update_extra_state()
    #
    #         self.local_step += 1
    #         self.global_step += 1
    #
    #         self.optimizer.zero_grad()
    #         # TODO 把loss_percep加进去
    #         # print(11111111111111111111111111111111111111111111111111111111111)
    #         with torch.cuda.amp.autocast(enabled=self.fp16):
    #             preds, truths, loss = self.train_step(data)
    #         # print('backwad')
    #         self.scaler.scale(loss).backward()
    #         # print('cut')
    #         # 针对性裁剪 loss1 对应的梯度
    #         self.scaler.unscale_(self.optimizer)
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #         # print('step')
    #         self.scaler.step(self.optimizer)
    #         # print(444444444444444444444444)
    #         self.scaler.update()
    #         # print(55555555555555555555555)
    #
    #         if self.scheduler_update_every_step:
    #             self.lr_scheduler.step()
    #
    #         loss_val = loss.item()
    #         total_loss += loss_val
    #
    #         if self.ema is not None and self.global_step % self.ema_update_interval == 0:
    #             self.ema.update()
    #
    #         if self.local_rank == 0:
    #             if self.report_metric_at_train:
    #                 for metric in self.metrics:
    #                     metric.update(preds, truths)
    #
    #             if self.use_tensorboardX:
    #                 self.writer.add_scalar("train/loss", loss_val, self.global_step)
    #                 self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
    #
    #             if self.scheduler_update_every_step:
    #                 pbar.set_description(
    #                     f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
    #             else:
    #                 pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
    #             pbar.update(loader.batch_size)
    #
    #     average_loss = total_loss / self.local_step
    #     self.stats["loss"].append(average_loss)
    #
    #     if self.local_rank == 0:
    #         pbar.close()
    #         if self.report_metric_at_train:
    #             for metric in self.metrics:
    #                 self.log(metric.report(), style="red")
    #                 if self.use_tensorboardX:
    #                     metric.write(self.writer, self.epoch, prefix="train")
    #                 metric.clear()
    #
    #     if not self.scheduler_update_every_step:
    #         if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #             self.lr_scheduler.step(average_loss)
    #         else:
    #             self.lr_scheduler.step()
    #
    #     self.log(f"==> Finished Epoch {self.epoch}.")
    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, mininterval=1,
                            bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0
        if self.opt.emo_lstm:
            self.model.emo_lstm_h0 = 0
            self.model.emo_lstm_c0 = 0
        for data in loader:
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, losses = self.train_step(data)

            loss = losses['total_loss']
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.ema is not None and self.global_step % self.ema_update_interval == 0:
                self.ema.update()

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)


                if self.use_tensorboardX and self.opt.torso == False:
                    self.writer.add_scalar("train/total_loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/perceptual_loss", losses['perceptual_loss'].item(), self.global_step)
                    self.writer.add_scalar("train/uncertainty_loss", losses['uncertainty_loss'].item(), self.global_step)
                    self.writer.add_scalar("train/lpips_loss", losses['lpips_loss'], self.global_step)
                    self.writer.add_scalar("train/eye_loss", losses['eye_loss'].item(), self.global_step)
                    self.writer.add_scalar("train/reg_loss", losses['reg_loss'], self.global_step)
                    self.writer.add_scalar("train/emo_loss", losses['emo_loss'].item(), self.global_step)
                    self.writer.add_scalar("train/loss_amb_aud", losses['loss_amb_aud'].item(), self.global_step)
                    self.writer.add_scalar("train/contrast_loss", losses['contrast_loss'].item(), self.global_step)
                    self.writer.add_scalar("train/weight", losses['weight'].item(), self.global_step)
                    self.writer.add_scalar("train/uncertainty", losses['uncertainty'].item(), self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                else:
                    self.writer.add_scalar("train/total_loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/perceptual_loss", losses['perceptual_loss'].item(), self.global_step)


                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")
    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, pred_ambient_aud, pred_ambient_eye, pred_ambient_emo, pred_uncertainty, truths, loss, loss_raw = self.eval_step(
                        data)

                # print(pred_ambient_emo.shape)
                # print(pred_ambient_eye.shape)  [1,512,512]
                # print(pred_ambient_emo)
                # print(pred_ambient_eye)
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    # for metric in self.metrics:
                    #     metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation',
                                                   f'{name}_{self.local_step:04d}_depth.png')
                    # save_path_error = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_errormap.png')
                    save_path_ambient_aud = os.path.join(self.workspace, 'validation',
                                                         f'{name}_{self.local_step:04d}_aud.png')
                    save_path_ambient_eye = os.path.join(self.workspace, 'validation',
                                                         f'{name}_{self.local_step:04d}_eye.png')
                    save_path_uncertainty = os.path.join(self.workspace, 'validation',
                                                         f'{name}_{self.local_step:04d}_uncertainty.png')
                    save_path_ambient_emo = os.path.join(self.workspace, 'validation',
                                                         f'{name}_{self.local_step:04d}_emo.png')
                    # save_path_gt = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_gt.png')

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    if self.opt.portrait:
                        pred = blend_with_mask_cuda(preds[0], data["bg_gt_images"].squeeze(0),
                                                    data["bg_face_mask"].squeeze(0))
                        preds = torch.from_numpy(pred).unsqueeze(0).to(self.device).float()
                    else:
                        pred = preds[0].detach().cpu().numpy()
                    pred_depth = preds_depth[0].detach().cpu().numpy()

                    for metric in self.metrics:
                        metric.update(preds, truths)

                        # loss_raw = loss_raw[0].mean(-1).detach().cpu().numpy()
                    # loss_raw = (loss_raw - np.min(loss_raw)) / (np.max(loss_raw) - np.min(loss_raw))
                    pred_ambient_aud = pred_ambient_aud[0].detach().cpu().numpy()
                    pred_ambient_aud /= np.max(pred_ambient_aud)
                    pred_ambient_eye = pred_ambient_eye[0].detach().cpu().numpy()
                    pred_ambient_eye /= np.max(pred_ambient_eye)
                    if self.opt.emo:
                        pred_ambient_emo = pred_ambient_emo[0].detach().cpu().numpy()
                        pred_ambient_emo /= np.max(pred_ambient_emo)
                    # print(1111111111111,pred_ambient_emo)

                    # pred_ambient = pred_ambient / 16
                    # print(pred_ambient.shape)
                    pred_uncertainty = pred_uncertainty[0].detach().cpu().numpy()
                    # pred_uncertainty = (pred_uncertainty - np.min(pred_uncertainty)) / (np.max(pred_uncertainty) - np.min(pred_uncertainty))
                    pred_uncertainty /= np.max(pred_uncertainty)

                    cv2.imwrite(save_path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    if not self.opt.torso:
                        cv2.imwrite(save_path_depth, (pred_depth * 255).astype(np.uint8))
                        # cv2.imwrite(save_path_error, (loss_raw * 255).astype(np.uint8))
                        cv2.imwrite(save_path_ambient_aud, (pred_ambient_aud * 255).astype(np.uint8))
                        cv2.imwrite(save_path_ambient_eye, (pred_ambient_eye * 255).astype(np.uint8))
                        if self.opt.emo:
                            cv2.imwrite(save_path_ambient_emo, (pred_ambient_emo * 255).astype(np.uint8))
                        cv2.imwrite(save_path_uncertainty, (pred_uncertainty * 255).astype(np.uint8))
                        # cv2.imwrite(save_path_gt, cv2.cvtColor((linear_to_srgb(truths[0].detach().cpu().numpy()) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        state['mean_count'] = self.model.mean_count
        state['mean_density'] = self.model.mean_density
        state['mean_density_torso'] = self.model.mean_density_torso

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                # always save new as best... (since metric cannot really reflect performance...)
                if True:

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded bare model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if 'mean_count' in checkpoint_dict:
            self.model.mean_count = checkpoint_dict['mean_count']
        if 'mean_density' in checkpoint_dict:
            self.model.mean_density = checkpoint_dict['mean_density']
        if 'mean_density_torso' in checkpoint_dict:
            self.model.mean_density_torso = checkpoint_dict['mean_density_torso']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


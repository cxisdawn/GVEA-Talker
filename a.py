import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from nerf_triplane.renderer import NeRFRenderer


class NeRFNetwork(nn.Module):
    def __init__(self,

                 audio_dim = 32,
                 # x=1
                 # torso net (hard coded for now)
                 ):
        super().__init__()


    def forward_torso(self, X, poses, c=None):
        # x: [N, 2] in [-1, 1]
        # head poses: [1, 4, 4]
        # c: [1, ind_dim], individual code

        # test: shrink x
        x = X * 0.8

        # deformation-based
        wrapped_anchor = self.anchor_points[None, ...] @ poses.permute(0, 2, 1).inverse()
        wrapped_anchor = (wrapped_anchor[:, :, :2] / wrapped_anchor[:, :, 3, None] / wrapped_anchor[:, :, 2, None]).view(1, -1)
        # print(wrapped_anchor)
        # enc_pose = self.pose_encoder(poses)
        enc_anchor = self.anchor_encoder(wrapped_anchor)
        enc_x = self.torso_deform_encoder(x)

        if c is not None:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1), c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1)], dim=-1)

        dx = self.torso_deform_net(h)

        x = (x + dx).clamp(-1, 1)

        x = self.torso_encoder(x, bound=1)

        # h = torch.cat([x, h, enc_a.repeat(x.shape[0], 1)], dim=-1)
        h = torch.cat([x, h], dim=-1)

        h = self.torso_net(h)

        alpha = torch.sigmoid(h[..., :1])*(1 + 2*0.001) - 0.001
        color = torch.sigmoid(h[..., 1:])*(1 + 2*0.001) - 0.001

        return alpha, color, dx
x = torch.randn(100, 2)
poses= torch.randn(1,4,4)

nf= NeRFNetwork(audio_dim=32,X=x,poses= poses)
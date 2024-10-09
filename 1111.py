# author time:2024-03-31
import torch
from scipy.spatial.transform import Slerp, Rotation
a= torch.randn(5,3,3)
b= Rotation.from_matrix(a[0:4]).mean().as_matrix()
print(a,a.shape)
print(b,b.shape)
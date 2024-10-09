# author time:2024-02-06
# perceptual loss function
import argparse
from fast_neural_style import functions as perceptual_functions
import torch
from torch import nn, Tensor
import fast_neural_style
import torchvision.transforms as transforms
# TV LOSS
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]





def main(opt):

    content_loss_func, style_loss_func = perceptual_functions.ContentLoss(opt.content_weight), perceptual_functions.StyleLoss(opt.style_weight)
    out = torch.randn(3, 512, 512)
    x = torch.randn(3, 512, 512)
    x2 = torch.randn(2,3, 512, 512)
    tvloss = TVLoss()

    print(tvloss.forward(x2),'11111')
    # transformed_x = fast_neural_style.utils.content_tensor_image_transform(512,x)
    # transformed_out = fast_neural_style.utils.content_tensor_image_transform(512,out)
    # print(transformed_x.shape)
    if opt.percep_loss_weight > 0:
        features_net = fast_neural_style.utils.features_extract_network(opt.arch, opt.content_layers, opt.style_layers)
        # print(features_net.features_layer)
        images_content_features, _ = features_net(x)
        transformed_content_features, transformed_style_features = features_net(out)
        content_loss = content_loss_func(transformed_content_features, images_content_features)
        print(content_loss)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #  Generic loss function options
    parser.add_argument('--tv_strength', default=1e-6)
    parser.add_argument('--percep_loss_weight', type=float, default=1.0)
    #  Options for feature reconstruction loss
    parser.add_argument('--content_weight', type=float,default=1.0)
    parser.add_argument("--content_layers", type=str, default="8", help="图片内容特征提取的层的索引序列")
    parser.add_argument("--content_size", type=int, default=256, help="训练图片的大小，默认为 256x256")

    # Options for style reconstruction loss
    parser.add_argument('--style_image', default='images/styles/candy.jpg')
    parser.add_argument("--style_size", type=int, default=512, help="风格图片的大小，默认为原图大小")
    parser.add_argument("--style_layers", type=str, default='-1' , help="图片风格特征提取的层的索引序列,[3,8,15,22]或[-1]")
    parser.add_argument("--style_weight", type=float, default=5.0, help="风格损失的权重")
    # parser.add_argument('--style_layers', default='4,9,16,23')
    parser.add_argument('--style_target_type', default='gram', choices=['gram', 'mean'])
    # vgg model
    parser.add_argument("--arch", type=str, default="vgg16", help="作图片特征提取的网络名称，目前支持的有 vgg16, vgg19")

    # Upsampling options
    parser.add_argument('--upsample_factor', type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=16, help="每一次训练迭代中数据的批量大小")
    parser.add_argument("--num_workers", type=int, default=4, help="并行读取数据的进程个数")

    opt = parser.parse_args()

    main(opt)
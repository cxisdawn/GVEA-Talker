import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from encoding import get_encoder
from .renderer import NeRFRenderer
import torch.nn.init as init


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


# Audio feature extractor
# 自注意力网络
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]  # [1,32,8]
        y = self.attentionConvNet(y)  # [1,1,8]
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)  # [1,8,1]
        return torch.sum(y * x, dim=1)  # [1, dim_aud]








# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size / 2)
        x = x[:, :, 8 - half_w:8 + half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)

        return out


# Audio feature extractor
class AudioNet_ave(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet_ave, self).__init__()
        # (n,512)
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, dim_aud),
        )

    def forward(self, x):
        # print(x.shape)  #(8,1,512)
        # half_w = int(self.win_size/2)
        # x = x[:, :, 8-half_w:8+half_w]
        # x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x).permute(1, 0, 2).squeeze(0)
        return x  # (8,32)实例化


class EmolabelEncoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=32):
        super(EmolabelEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x  # (1,32)


# todo dim marching during instantiation
class ContrastiveNet(nn.Module):
    def __init__(self, audio_dim=32,projection_dim = 1024):
        super(ContrastiveNet, self).__init__()
        # 定义投影层，只对音频特征进行投影
        self.aud_proj = nn.Sequential(
            nn.Linear(audio_dim, 256),  # 从 32 维映射到 256 维
            nn.ReLU(),
            nn.Linear(256, 512),  # 从 32 维映射到 256 维
            nn.ReLU(),           # 非线性激活函数
            nn.Linear(512, projection_dim) # 从 256 维映射到 1024 维
        )



    def forward(self, enc_a, enc_b):
        # enc_a [1,32]
        # 对音频特征进行投影
        out1 = self.aud_proj(enc_a)
        out2 = self.aud_proj(enc_b)
        # 归一化，以便使用余弦相似度
        out1 = F.normalize(out1, p=2, dim=1)
        out2 = F.normalize(out2, p=2, dim=1)
        # print(out1.shape) [1,1024]
        return out1, out2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5 ):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


    def forward(self, audact_proj, audneg_proj, emo_feat,):
        # 归一化 emo_feat
        emo_feat = F.normalize(emo_feat, p=2, dim=1)
        # 计算余弦相似度
        pos_similarity = F.cosine_similarity(audact_proj, emo_feat)
        neg_similarity = F.cosine_similarity(audact_proj, audneg_proj)

        # 对比损失：拉近 audact 和 emo_feat，拉远 audact 和 audneg
        loss = torch.mean(torch.clamp(self.margin - pos_similarity + neg_similarity, min=0.0))
        # print(loss)
        return loss
# mamba注意力网络
class AudioMamba2Model(torch.nn.Module):
    def __init__(self, in_dim=36 ,aud_dim=32 ,d_state=4,d_conv=2,expand=3):
        super(AudioMamba2Model, self).__init__()

        from mamba_ssm import Mamba2
        self.in_dim = in_dim
        self.aud_dim = aud_dim
        self.mamba = Mamba2(
            d_model=aud_dim,  # 输入和输出维度
            d_state=8,  # SSM状态扩展因子
            d_conv=2,  # 局部卷积宽度
            expand=2,  # 块扩展因子
            headdim = 8
        )



        # 当当前维度大于1时，继续添加层
        if in_dim != aud_dim:

            self.f_out = nn.Linear(in_dim, aud_dim)


        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape,111111111111111111111111111111111111111111111)
        # print(x.shape) # [N,36]
        # x的形状为[batch_size, sequence_length, aud_dim]
        x = x.unsqueeze(1)
        # print(x.shape,33333333333333333333)
        if self.in_dim != self.aud_dim:
            x = self.f_out(x)
        print(x.shape)
        y = self.mamba(x).squeeze(1)  # [N,36]
        # y = self.attentionNet(y)  # [1,8,1]
        out = self.softmax(y)  # 在这里应用softmax
        # print('out    11111111111111111111111111')
        # y= nn.Softmax(dim=1)
        # print(out.shape)
        return out # [1, dim_aud]

        # # 将输出转换为所需的形状[1, aud_dim]
        # y = y.mean(dim=1, keepdim=False)
        # return y

# mamba注意力网络
class AudioMamba1Model(torch.nn.Module):
    def __init__(self, in_dim=36 ,aud_dim=32 ,d_state=4,d_conv=2,expand=3):
        super(AudioMamba1Model, self).__init__()
        from mamba_ssm import Mamba
        self.in_dim = in_dim
        self.hidden_size = 8
        self.aud_dim = aud_dim
        self.mamba = Mamba(
            d_model=self.hidden_size,  # 输入和输出维度
            d_state=d_state,  # SSM状态扩展因子
            d_conv=d_conv,  # 局部卷积宽度
            expand=expand,  # 块扩展因子
        )

        self.f_in = nn.Linear(in_dim, self.hidden_size)

        # if in_dim != aud_dim:

        self.f_out = nn.Linear(self.hidden_size, aud_dim)


        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape) # [N,36]
        # x的形状为[batch_size, sequence_length, aud_dim]
        x = x.unsqueeze(1)    #[N,36]-->[N,1,36]
        y_in = self.f_in(x)   # [N,1,36] --> [N,1,8]

        y = self.mamba(y_in).squeeze(1)  # [N,8]


        out = self.f_out(y)   # [N,32]
        out = self.softmax(out)  # 在这里应用softmax

        #print(out.shape)
        return out # [1, dim_aud]

        # # 将输出转换为所需的形状[1, aud_dim]
        # y = y.mean(dim=1, keepdim=False)
        # return y

# class AudioMamba1Model(torch.nn.Module):
#     def __init__(self, in_dim=36 ,aud_dim=32 ,d_state=4,d_conv=2,expand=3):
#         super(AudioMamba1Model, self).__init__()
#         from mamba_ssm import Mamba
#         self.in_dim = in_dim
#         self.hidden_size = 8
#         self.aud_dim = aud_dim
#         self.mamba = Mamba(
#             d_model=self.in_dim,  # 输入和输出维度
#             d_state=d_state,  # SSM状态扩展因子
#             d_conv=d_conv,  # 局部卷积宽度
#             expand=expand,  # 块扩展因子
#         )
#         # if in_dim != aud_dim:
#
#         self.f_out = nn.Linear(in_dim, aud_dim)
#
#
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         # print(x.shape) # [N,36]
#         # x的形状为[batch_size, sequence_length, aud_dim]
#         x = x.unsqueeze(1)    #[N,36]-->[N,1,36]
#
#         y = self.mamba(x).squeeze(1)  # [N,36]
#
#
#         out = self.f_out(y)   # [N,32]
#         out = self.softmax(out)  # 在这里应用softmax
#
#         #print(out.shape)
#         return out # [1, dim_aud]
#
#         # # 将输出转换为所需的形状[1, aud_dim]
#         # y = y.mean(dim=1, keepdim=False)
#         # return y

# TODO mlp网络
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):  # out_dim=32
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)

        return x


# class EmoMLP(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_hidden, num_layers):  # out_dim=32
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_out = dim_out
#         self.dim_hidden = dim_hidden
#         self.num_layers = num_layers
#
#         net = []
#         for l in range(num_layers):
#             net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
#                                  self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))
#
#         self.net = nn.ModuleList(net)
#
#     def forward(self, x):
#         for l in range(self.num_layers):
#             x = self.net[l](x)
#             if l != self.num_layers - 1:
#                 x = F.relu(x)
#                 # x = F.dropout(x, p=0.1, training=self.training)
#
#         return x
class EmoMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            layer = nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                              self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False)
            # He 初始化
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
            net.append(layer)

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x)
                # x = F.dropout(x, p=0.1, training=self.training)

        return x

# todo 进一步学习注意力网络
#         利用transformers

def swish(x):
    return x * torch.sigmoid(x)


class ResLine(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResLine, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.extra_fc = nn.Linear(hidden_dim, hidden_dim)  # 添加额外的线性层
        # 对权重进行 He 初始化
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.extra_fc.weight)

    def forward(self, x):
        residual = x
        # out = F.relu(self.fc1(x))
        # out = self.fc2(out)
        # out2 = self.fc3()
        out = swish(self.fc1(x))  # 使用 Swish 激活函数
        out = swish(self.fc2(out))  # 使用 Swish 激活函数
        out = swish(self.extra_fc(out))  # 使用 Swish 激活函数
        # out = F.gelu(self.fc1(x))  # 使用 GELU 激活函数
        # out = F.gelu(self.fc2(out))  # 使用 GELU 激活函数
        # out = F.gelu(self.extra_fc(out))  # 使用 GELU 激活函数
        out += 0.5 * residual
        return out

        # out += residual
        # return out


class LineModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=1):
        super(LineModel, self).__init__()
        self.mlp = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([ResLine(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.final_fc = nn.Linear(hidden_dim, output_dim)
        # 对 MLP 层的权重进行 He 初始化
        nn.init.kaiming_normal_(self.mlp.weight)

        # 对最终全连接层的权重进行 He 初始化
        nn.init.kaiming_normal_(self.final_fc.weight)

    def forward(self, x):
        # print(x.shape) # [1048576, 36]
        x = F.elu(self.mlp(x))
        for block in self.res_blocks:
            x = block(x)
        x = self.final_fc(x)
        return x





# transform torso
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print('x.shape',x.shape) # [65536, 84]
        x = x + self.pe[:x.size(0), :x.size(1)].requires_grad_(False)
        return self.dropout(x)


# class TorNet(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, d_k, d_v, d_model, d_ff, pad_idx, max_len=128):
#         super(TorNet, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
#         self.positional_encoding = PositionalEncoding(embedding_dim, max_len=max_len)
#         self.transformer_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=0.1)
#             for _ in range(num_layers)
#         ])
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#
#     def forward(self, src_seq):
#
#         embedded = self.embedding(src_seq)
#         embedded = self.positional_encoding(embedded)
#         encoded = embedded.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
#         for layer in self.transformer_layers:
#             encoded = layer(encoded)
#         encoded = self.layer_norm(encoded)
#         return encoded
# 构建Transformer模型
class TorNet(nn.Module):
    def __init__(self, input_size, output_size, mlp_layers=2, num_layers=4, hidden_size=512, num_heads=4, dropout=0.1):
        super(TorNet, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                        dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        self.fc3 = nn.Linear(input_size // 4, output_size)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = x.permute(1, 0, 2)  # (seq_len 1, batch_size 65536 , embedding_dim 84)
        x = self.transformer_encoder(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], -1)  # (batch_size 65536 , embedding_dim 2)
        return x


# todo Emotionalnet
class EmotionalNet(nn.Module):
    def __init__(self, audio_dim=32, hidden_size=64, lstm_num_layers=2):
        """
        :param audio_dim:
                h0和c0的维度：(num_layers * num_directions, batch, hidden_size)
                        num_directions 是LSTM的方向数，通常为1（单向）或2（双向）
                        lstm的输入：(batch_size,T, input_size)
        """

        super(EmotionalNet, self).__init__()
        # 定义网络层
        # self.lstm_x = nn.LSTM(input_size=32, hidden_size=64)
        self.lstm_a = nn.LSTM(input_size=audio_dim, hidden_size=hidden_size, num_layers=lstm_num_layers)
        # self.lstm_e = nn.LSTM(input_size=1, hidden_size=64)
        # self.dense = nn.Linear(in_features=192, out_features=64)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, enc_x, enc_a, e, h0, c0):
        # enc_a [1,32]
        # enc_x [N,32]
        # LSTM层处理输入
        # lstm_x_out, _ = self.lstm_x(enc_x)
        if h0 is None:
            h0 = self.h_init()
        if c0 is None:
            c0 = self.c_init()
        lstm_a_out, h0, c0 = self.lstm_a(enc_a, (h0, c0))
        # lstm_e_out, _ = self.lstm_e(e)

        # 合并所有LSTM层的输出
        # merged = torch.cat((lstm_x_out[-1], lstm_a_out[-1], lstm_e_out[-1]), dim=1)

        # 全连接层和输出层
        # dense_out = self.dense(merged)
        emo = torch.sigmoid(self.output_layer(lstm_a_out))
        return emo, h0, c0

    def h_init(self, ):
        # 初始化隐藏状态为零
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                )

    def c_init(self, ):
        # 初始化细胞状态为零
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                )


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 audio_dim=32,
                 # torso net (hard coded for now)
                 ):
        super().__init__(opt)

        # audio embedding
        self.emb = self.opt.emb

        # #是否用对比学习
        # self.contrast_loss =self.opt.contrast_loss
        # if self.contrast_loss:
        #     self.ContrastNet=ContrastiveNet()
        #     self.contrastloss=ContrastiveLoss(margin=0.1)
        # else:
        #     self.contrastloss=0
        #
        #

        if 'esperanto' in self.opt.asr_model:
            self.audio_in_dim = 44
        elif 'deepspeech' in self.opt.asr_model:
            self.audio_in_dim = 29
        elif 'hubert' in self.opt.asr_model:
            self.audio_in_dim = 1024
        else:
            self.audio_in_dim = 32

        if self.emb:
            self.embedding = nn.Embedding(self.audio_in_dim, self.audio_in_dim)

        # audio network
        self.audio_dim = audio_dim
        if self.opt.asr_model == 'ave':
            # print('use_ave')
            # (8,1,512)
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        # self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.att = self.opt.att
        # 音频注意力机制网络

        if self.att > 0 and self.opt.use_atan is False:
            self.audio_att_net = AudioAttNet(self.audio_dim)

        elif self.opt.use_atan == 'AudioMambaModel':
            self.audio_att_net = AudioMambaModel(self.audio_dim)
        # seq_len = 128  # 音频特征在时间维度上的长度
        # dim_aud = 512  # 音频特征的维度
        # nhead = 8  # 多头注意力机制中的头数
        # num_encoder_layers = 3  # 编码器层的数量
        # dim_feedforward = 2048  # 前馈网络的维度
        #
        #a = AudioMambaModel(self.audio_dim)
        # self.mamba = Mamba(
        #     d_model=self.audio_dim,  # 输入和输出维度
        #     d_state=4,  # SSM状态扩展因子
        #     d_conv=2,  # 局部卷积宽度
        #     expand=2,  # 块扩展因子
        # )
        # # 创建模型
        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=64,
                                                      log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=64,
                                                      log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=64,
                                                      log2_hashmap_size=14, desired_resolution=512 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 64
        self.geo_feat_dim = 64
        self.eye_att_net = MLP(self.in_dim, 1, 16, 2)

        # self.eye_att_net = MLP(self.in_dim, 1, 16, 2)
        # self.eye_dim = 1 if self.exp_eye else 0

        self.contrast_loss = self.opt.contrast_loss
        if self.contrast_loss:
            self.contrastNet = ContrastiveNet(audio_dim=self.audio_dim,projection_dim=1024)
            self.contrastloss = ContrastiveLoss()
        else:
            pass

        # TODO Synctalk的眼睛模型
        if self.opt.au45:
            self.eye_att_net = MLP(self.in_dim, 1, 16, 2)
            self.eye_dim = 1 if self.exp_eye else 0
        else:
            if self.opt.bs_area == "upper":
                self.eye_att_net = MLP(self.in_dim, 7, 64, 2)
                self.eye_dim = 7 if self.exp_eye else 0
            elif self.opt.bs_area == "single":
                self.eye_att_net = MLP(self.in_dim, 4, 64, 2)
                self.eye_dim = 4 if self.exp_eye else 0
            elif self.opt.bs_area == "eye":
                self.eye_att_net = MLP(self.in_dim, 2, 64, 2)
                self.eye_dim = 2 if self.exp_eye else 0

        self.emo_one_hot_dim = 9 if self.opt.emo else 0
        # print(self.in_dim + self.audio_dim + self.eye_dim + self.emo_one_hot_dim,11111111111111111111111111111111111111)
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.emo_one_hot_dim, 1 + self.geo_feat_dim,
                             self.hidden_dim,
                             self.num_layers)
        # TODO emotion网络
        # if self.opt.emo_lstm == True:
        #     self.emo_lstm_h0 = None
        #     self.emo_lstm_c0 = None
        #     self.emotional_net = EmotionalNet(self.audio_dim)  # 预测情绪的网络
        if self.opt.emo:
            self.emo_att_net = EmoMLP(self.in_dim, 1, 16, 2)  # 情绪的注意力网络
        ## color network
        self.num_layers_color = 2
        self.hidden_dim_color = 64
        self.encoder_dir, self.in_dim_dir = get_encoder('spherical_harmonics')
        self.color_net = MLP(self.in_dim_dir + self.geo_feat_dim + self.individual_dim, 3, self.hidden_dim_color,
                             self.num_layers_color)

        self.unc_net = MLP(self.in_dim, 1, 32, 2)


        # 示例参数
        # growth_rate = 32
        # num_layers = 6
        # input_dim = 64
        # hidden_dim = 128
        # output_dim = 10
        # 音频对采样点的注意力网络 输入是in_dim 输出是audio_dim
        if self.opt.aud_ch_att_net == 'MLP':
            self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 64, 2)  # audio_dim=32
        elif self.opt.aud_ch_att_net == 'LineModel': # 残差块
            self.aud_ch_att_net = LineModel(self.in_dim, 128, self.audio_dim)
        elif self.opt.aud_ch_att_net == 'Mamba2':
            self.aud_ch_att_net = AudioMamba2Model(in_dim=self.in_dim,aud_dim = self.audio_dim,d_state=4,d_conv=2,expand=2)
        elif self.opt.aud_ch_att_net == 'Mamba1':
            self.aud_ch_att_net = AudioMamba1Model(in_dim=self.in_dim,aud_dim = self.audio_dim,d_state=4,d_conv=2,expand=2)

        self.testing = False

        # ---------------------------------------

        # todo network torso process
        if self.torso:  # 如果定义了躯干部分
            # 躯干形变网络
            self.register_parameter('anchor_points',
                                    nn.Parameter(torch.tensor(
                                        [[0.01, 0.01, 0.1, 1], [-0.1, -0.1, 0.1, 1], [0.1, -0.1, 0.1, 1]])))
            # 注册锚点参数，这些是用于形变网络的基准点

            self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('frequency', input_dim=2, multires=8)
            # 获取一个频率编码器，用于躯干形变，输入维度为2，多分辨率级别为8

            # self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=1, base_resolution=16, log2_hashmap_size=16, desired_resolution=512)
            # 这是另一种编码器的设置，但被注释掉了

            self.anchor_encoder, self.anchor_in_dim = get_encoder('frequency', input_dim=6, multires=3)
            # 获取一个频率编码器，用于锚点，输入维度为6，多分辨率级别为3
            if self.opt.tornet == False:
                self.torso_deform_net = MLP(self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso,
                                            2, 32, 3)
            # 定义一个多层感知机（MLP），用于躯干形变，输入维度是躯干形变编码器的维度加上锚点编码器的维度加上躯干的个体维度，输出维度为2，隐藏层维度为32，隐藏层数量为3
            else:
                self.torso_deform_net = TorNet(input_size=84, output_size=1, mlp_layers=2)

            # 躯干颜色网络
            self.torso_encoder, self.torso_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=2,
                                                                base_resolution=16, log2_hashmap_size=16,
                                                                desired_resolution=2048)
            # 获取一个平铺网格编码器，用于躯干颜色，输入维度为2，多级别为16，每级别维度为2，基础分辨率为16，哈希表大小的对数为16，期望分辨率为2048
            self.torso_net = MLP(
                self.torso_in_dim + self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso, 4, 32, 3)
            # 定义一个多层感知机（MLP），用于躯干颜色，输入维度是躯干编码器的维度加上躯干形变编码器的维度加上锚点编码器的维度加上躯干的个体维度，输出维度为4，隐藏层维度为32，隐藏层数量为3

    # -----------------------------------------------# x= torch.randn(100, 2)
    def forward_torso(self, x, poses, c=None):
        # x: [N, 2] in [-1, 1]
        # head poses: [1, 4, 4]
        # c: [1, ind_dim], individual code

        # test: shrink x
        x = x * self.opt.torso_shrink

        # deformation-based
        wrapped_anchor = self.anchor_points[None, ...] @ poses.permute(0, 2, 1).inverse()
        wrapped_anchor = (
                wrapped_anchor[:, :, :2] / wrapped_anchor[:, :, 3, None] / wrapped_anchor[:, :, 2, None]).view(1,
                                                                                                               -1)
        # print(wrapped_anchor)
        # enc_pose = self.pose_encoder(poses)
        enc_anchor = self.anchor_encoder(wrapped_anchor)
        enc_x = self.torso_deform_encoder(x)

        if c is not None:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1), c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1)], dim=-1)

        # print(h.dtype,h.shape) [[16384, 84],[65536, 84]之间变动]
        dx = self.torso_deform_net(h)

        x = (x + dx).clamp(-1, 1)

        x = self.torso_encoder(x, bound=1)

        # h = torch.cat([x, h, enc_a.repeat(x.shape[0], 1)], dim=-1)
        h = torch.cat([x, h], dim=-1)

        h = self.torso_net(h)

        alpha = torch.sigmoid(h[..., :1]) * (1 + 2 * 0.001) - 0.001
        color = torch.sigmoid(h[..., 1:]) * (1 + 2 * 0.001) - 0.001

        return alpha, color, dx

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:, :1], x[:, -1:]], dim=-1)
        return xy, yz, xz

    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)

        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)

    # [N, D_xy + D_yz + D_xz]

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        # print(a.shape)      # [8,1,512]
        if self.emb:
            a = self.embedding(a).transpose(-1, -2).contiguous()  # [1/8, 29, 16]

        enc_a = self.audio_net(a)  # [1/8, 64]  #now [8,32]
        # print(enc_a.shape)   # ave [8,32]
        # print(enc_a.shape,1111111111111111111111111111233)
        if self.att > 0:
            enc_a = self.audio_att_net(enc_a.unsqueeze(0))  # [1, 64] # now [1,32]

        # print(enc_a.shape,1111111111111111111111111111233) # result mamba[1,32]

        return enc_a

    def predict_uncertainty(self, unc_inp):
        if self.testing or not self.opt.unc_loss:
            unc = torch.zeros_like(unc_inp)
        else:
            unc = self.unc_net(unc_inp.detach())

        return unc

    def forward(self, x, d, enc_a, c, e=None, emo_label=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # enc_a: [1, aud_dim]
        # c: [1, ind_dim], individual code
        # e: [1, 1], eye feature

        # TODO  adding emo_label
        # emo_label[B,1] => [1,1]

        enc_x = self.encode_x(x, bound=self.bound)  # [1048576, 36]
        # print(enc_x.shape,'1234')

        sigma_result = self.density(x, enc_a, e, enc_x, emo_label)
        sigma = sigma_result['sigma']
        geo_feat = sigma_result['geo_feat']
        aud_ch_att = sigma_result['ambient_aud']
        eye_att = sigma_result['ambient_eye']

        emo_att = sigma_result['ambient_emo']

        # color
        enc_d = self.encoder_dir(d)

        if c is not None:
            h = torch.cat([enc_d, geo_feat, c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_d, geo_feat], dim=-1)

        h_color = self.color_net(h)
        color = torch.sigmoid(h_color) * (1 + 2 * 0.001) - 0.001

        uncertainty = self.predict_uncertainty(enc_x)
        uncertainty = torch.log(1 + torch.exp(uncertainty))
        #####
        # if self.opt.use_emo_loss:
        #     emo_att = sigma_result['ambient_emo']
        #     return sigma, color, aud_ch_att, eye_att, emo_att,uncertainty[..., None]
        return sigma, color, aud_ch_att, eye_att, emo_att, uncertainty[..., None]

    def density(self, x, enc_a, e=None, enc_x=None, emo_label=None):
        # x: [N, 3], in [-bound, bound]
        # print(x.shape)
        # enc_a 已经编码过的音频特征 [1,32]
        # 对采样点进行编码
        if enc_x is None:
            enc_x = self.encode_x(x, bound=self.bound)  # enc_x.shape[N, D_xy + D_yz + D_xz]
        # enc_a   (1,32)
        # 每个采样点复制一份音频特征
        enc_a = enc_a.repeat(enc_x.shape[0], 1)    #[2097152,32]  采样点的数量]

        # aud_ch_att = self.aud_ch_att_net(enc_x)
            # print(enc_x.shape)  #[2097152,36]  采样点的数量

        aud_ch_att = self.aud_ch_att_net(enc_x)   # aud_ch_att_net 注意力网络
        enc_w = enc_a * aud_ch_att

        if e is not None:
            # e = self.encoder_eye(e)
            eye_att = torch.sigmoid(self.eye_att_net(enc_x))
            e = e * eye_att
            # e = e.repeat(enc_x.shape[0], 1)  [N,1]
            h = torch.cat([enc_x, enc_w, e], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w], dim=-1)

        # TODO 情绪注意力
        if emo_label is not None:
            
            emo_att = torch.sigmoid(self.emo_att_net(enc_x))
            # print(emo_att.size()) [N,1]

            # # 找到值为1的位置
            # position = torch.argmax(emo_label)
            # # print( position + 1)
            # emo = (position - 1) / (9 - 1) * emo_att
            # print('11111',emo_label) [1,9]
            emo = emo_label * emo_att
            # print(emo)
            # print(emo.shape)  [N,9]
            h = torch.cat([h, emo], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w, e], dim=-1)
            emo_att = torch.zeros_like(eye_att)
            # print(emo.shape) #[1048576]
        # 密度网络
        h = self.sigma_net(h)
        # 拿出密度
        sigma = torch.exp(h[..., 0])
        # 用于颜色网络的总体特征
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'ambient_aud': aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye': eye_att.norm(dim=-1, keepdim=True),
            'ambient_emo': emo_att,

        }

    def contrast(self, audact, audneg, emo_feat):
        a = self.audio_net(audact)  # [8,32]
        # print(a.shape)
        enc_a = self.audio_att_net(a.unsqueeze(0))

        b = self.audio_net(audneg)
        enc_b = self.audio_att_net(b.unsqueeze(0))
        output_act, output_neg = self.contrastNet(enc_a, enc_b)
        contrast_loss = self.contrastloss(output_act, output_neg, emo_feat)
        return contrast_loss


        # # 一帧的情况下
        # # audact [1,1,512]
        # a = self.audio_net(audact)
        # # print(a.shape)
        # enc_a = a.unsqueeze(0)
        #
        # b = self.audio_net(audneg)
        # enc_b = b.unsqueeze(0) # [1,32]
        # #print(enc_b.shape)
        # output_act, output_neg = self.contrastNet(enc_a, enc_b)
        # contrast_loss = self.contrastloss(output_act, output_neg, emo_feat)
        # return contrast_loss

    # usage: optimizer parameters
    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):
        # wd 1e-3
        # ONLY train torso
        if self.torso:
            params = [
                {'params': self.torso_encoder.parameters(), 'lr': lr},
                {'params': self.torso_deform_encoder.parameters(), 'lr': lr, 'weight_decay': wd},
                {'params': self.torso_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.torso_deform_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.anchor_points, 'lr': lr_net, 'weight_decay': wd}
            ]

            if self.individual_dim_torso > 0:
                params.append({'params': self.individual_codes_torso, 'lr': lr_net, 'weight_decay': wd})

            return params

        params = [  # all parameters
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd},

            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            # {'params': self.encoder_xyz.parameters(), 'lr': lr},

            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        if self.att > 0:
            params.append(
                {'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})  # 学习率调小 权重衰减 调大
        if self.emb:
            params.append({'params': self.embedding.parameters(), 'lr': lr})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        if self.train_camera:
            params.append({'params': self.camera_dT, 'lr': 1e-5, 'weight_decay': 0})
            params.append({'params': self.camera_dR, 'lr': 1e-5, 'weight_decay': 0})


        params.append({'params': self.unc_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        # if self.opt.emo_lstm:
        #     params.append({'params': self.emotional_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        if self.opt.emo:
            # 将网络加进训练
            params.append({'params': self.emo_att_net.parameters(), 'lr': lr_net * 0.05, 'weight_decay': wd})
        if self.opt.contrast_loss:
            params.append({'params': self.contrastNet.parameters(), 'lr': lr_net * 0.05, 'weight_decay': wd})
        
        if self.opt.aud_ch_att_net=="Mamba1":
            # print("mamba122222222222222222222222222222222222222222222222222222222222222222222222222222222222222")
            params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})   
        else:
            # print("11111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
            params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        return params


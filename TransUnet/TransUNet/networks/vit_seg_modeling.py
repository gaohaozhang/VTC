# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}#这一段是什么，可以改吗


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis# store whether visualization is required or not
        self.num_attention_heads = config.transformer["num_heads"]# get the number of attention heads from the configuration
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)# calculate the size of each attention head
        self.all_head_size = self.num_attention_heads * self.attention_head_size # calculate the total size of all the attention heads

        self.query = Linear(config.hidden_size, self.all_head_size) # create linear layers to transform the input hidden states for query, key, and value
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)# create a linear layer to transform the attention output
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])# create dropout layers for attention and projection
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)# create a softmax layer to calculate attention scores

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)# reshape the input tensor for attention calculation
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)# transpose the tensor to match the expected shape

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)# transform the hidden_states tensor for query, key, and value
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)# Transpose the output of linear transformation for query, key, and value layers
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))# calculate the attention scores using matrix multiplication and scaling by sqrt(attention_head_size)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores) # apply softmax to calculate attention probabilities
        weights = attention_probs if self.vis else None# If visualization is required, return the attention probabilities
        attention_probs = self.attn_dropout(attention_probs)# Apply dropout to attention probabilities

        context_layer = torch.matmul(attention_probs, value_layer) # Calculate the weighted sum of value layer using attention probabilities to get the context layer
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()# Permute the dimensions of the context layer and reshape it to the desired shape
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer) # Apply linear transformation to the context layer to get the attention output
        attention_output = self.proj_dropout(attention_output) # Apply dropout to the attention output
        return attention_output, weights # Return the attention output and attention probabilities (if visualization is required)


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])# 创建一个线性层，将config.hidden_size个输入特征映射到config.transformer["mlp_dim"]个输出特征
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size) # 创建另一个线性层，将config.transformer["mlp_dim"]个输入特征映射到config.hidden_size个输出特征
        self.act_fn = ACT2FN["gelu"]# 将激活函数设置为高斯误差线性单元（GELU）函数
        self.dropout = Dropout(config.transformer["dropout_rate"])# 创建一个dropout层，dropout率设置为config.transformer["dropout_rate"]

        self._init_weights() # 调用私有方法_init_weights()，该方法用于初始化fc1和fc2层的权重参数

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)# 初始化fc1层的权重参数，使用Xavier均匀分布初始化方法
        nn.init.xavier_uniform_(self.fc2.weight) # 初始化fc2层的权重参数，使用Xavier均匀分布初始化方法
        nn.init.normal_(self.fc1.bias, std=1e-6)# 初始化fc1层的偏置项，使用均值为0，标准差为1e-6的正态分布随机初始化方法
        nn.init.normal_(self.fc2.bias, std=1e-6)# 初始化fc2层的偏置项，使用均值为0，标准差为1e-6的正态分布随机初始化方法

    def forward(self, x):
        x = self.fc1(x)# 将输入x传递给fc1层，得到输出
        x = self.act_fn(x) # 将输出x传递给激活函数，得到激活后的输出
        x = self.dropout(x) # 将激活后的输出x传递给dropout层，得到随机失活后的输出
        x = self.fc2(x)# 将随机失活后的输出x传递给fc2层，得到最终的输出
        x = self.dropout(x) # 将最终的输出x再次传递给dropout层，进行最后的随机失活处理
        return x# 返回最终的输出


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)     # 将输入的图像尺寸转换为元组形式，方便后续使用

        if config.patches.get("grid") is not None:   # # 判断模型是否为 ResNet
            grid_size = config.patches["grid"] # 从配置文件中获取 patch 的网格大小
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])# 计算 patch 的大小
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16) # 实际的 patch 大小
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])   # 计算图像中 patch 的数量
            self.hybrid = True# 标记当前模型为混合模型
        else: # 对于 ViT 模型
            patch_size = _pair(config.patches["size"])# 从配置文件中获取 patch 的大小
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) # 计算图像中 patch 的数量
            self.hybrid = False# 标记当前模型为非混合模型

        if self.hybrid: # 如果当前模型是混合模型
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)# 定义 ResNetV2 模型
            in_channels = self.hybrid_model.width * 16 # 获取 ResNetV2 模型的输出通道数
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)# 定义 patch 的卷积层
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size)) # 定义位置嵌入矩阵

        self.dropout = Dropout(config.transformer["dropout_rate"])# 定义 dropout 层


    def forward(self, x):
        if self.hybrid: # 如果模型是混合模型，调用hybrid_model函数并获取输出
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)   # 将 patch embeddings 展平成一个向量 (B, hidden, n_patches)
        x = x.flatten(2) # 将 patch embeddings 展平成一个向量 (B, hidden, n_patches)
        x = x.transpose(-1, -2)  # 将展平后的 patch embeddings 变形，变成 (B, n_patches, hidden)

        embeddings = x + self.position_embeddings# 为 patch embeddings 添加位置嵌入
        embeddings = self.dropout(embeddings)# 对 patch embeddings 进行 dropout 处理
        return embeddings, features # 返回 patch embeddings 和 features


class CBAM(nn.Module):
    def __init__(self, channel, reduction=4, spatial_kernel=3):
        super(CBAM, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size#从config对象中获取hidden_size的值，并将其存储在实例变量中。
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)#使用LayerNorm类创建一个层归一化器，其中config.hidden_size是层归一化的大小，eps是一个很小的值，以避免分母为0。
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)#使用Mlp类创建一个多层感知机。
        self.attn = Attention(config, vis)

        self.convReLu = Conv2dReLU(in_channels=self.hidden_size,out_channels=self.hidden_size,kernel_size=1,padding=0)
        self.cbam = CBAM(channel=self.hidden_size,reduction=4,spatial_kernel=3)#reduction是用于计算通道权重的线性层的缩小因子，spatial_kernel是计算空间权重的卷积核大小。



    def forward(self, x):
        #h = x
        #print('179',x.shape)
        z = x#x为图片参数，大小为(batch_size，num_channels，image_size)。

        b,z1,z2 = z.size()#b, z1, z2 分别代表 batch size、input channels、sequence length（或者 patch 数量）
        z3 = int(math.sqrt(z1))#计算z3作为z1的平方根，转换为整数。
        z = z.transpose(-1,-2)#z被转置，使得最后两个维度交换。
        #print('180',z.shape)
        z = z.reshape(b,z2,z3,z3)#然后将z重新形状为(batch_size，z2，z3，z3)的尺寸。
        z = self.convReLu(z)
        z = self.cbam(z)
        z = z.flatten(2)
        z = z.transpose(-1, -2)
        '''重塑后的张量z通过卷积层和ReLU激活函数传递。

在卷积层的输出上调用cbam方法，以应用通道注意力和空间注意力机制。

输出张量沿最后一个维度被展平。

展平的张量被转置，使得最后两个维度交换。

最终张量被作为forward函数的输出返回。'''

        #print('184', z.shape)
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        # x = x + h
        x = x + z#对输入张量 x 进行归一化处理，然后应用自注意力机制，得到注意力权重张量 weights。最后将 z 与 x 相加，得到融合后的特征张量。

        h = x
        x = self.ffn_norm(x)#将融合后的特征张量赋值给变量 h，并对其进行归一化处理。
        x = self.ffn(x)
        x = x + h#然后应用前馈神经网络（Feedforward Neural Network，FFN）进行特征变换，再将其与 h 相加。
        return x, weights#返回更新后的特征张量 x 和注意力权重张量 weights。

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"#存放权重文件的路径
        with torch.no_grad():#使用torch.no_grad()上下文，以确保在加载权重时不会计算梯度。
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
#从预训练权重中提取查询、键、值和输出权重，将它们转换为PyTorch张量，并调整形状以匹配模型权重。
            #从预训练权重中提取查询、键、值和输出偏置，将它们转换为PyTorch张量，并调整形状以匹配模型偏置。
            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
#. 将查询、键、值和输出权重复制到模型的注意力层。
            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)
#从预训练权重中提取前馈网络的第一个和第二个线性层的权重和偏置，将它们转换为PyTorch张量，并调整形状以匹配模型权重和偏置。
            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
#将注意力规范化层的权重和偏置复制到模型的注意力规范化层。
            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)
#从预训练权重中提取前馈网络规范化层的权重和偏置，将它们转换为PyTorch张量。将前馈网络规范化层的权重和偏置复制到模型的前馈网络规范化层。

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()#调用父类构造函数super(Encoder, self).__init__()。
        self.vis = vis
        self.layer = nn.ModuleList()#初始化一个名为self.layer的nn.ModuleList，用于存储Transformer的编码器层。
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))#在一个循环中，创建config.transformer["num_layers"]个编码器层，并将它们添加到self.layer列表中。

    def forward(self, hidden_states):
        attn_weights = []#初始化一个名为attn_weights的空列表，用于存储注意力权重
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)#循环处理self.layer列表中的每个layer_block，并更新hidden_states和attn_weights
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
#对编码器的输出hidden_states应用self.encoder_norm 返回经过编码器处理的encoded以及注意力权重attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)# 创建 Transformer 实例
        self.decoder = DecoderCup(config)# 创建 DecoderCup 实例
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )# 创建 SegmentationHead 实例
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)# 若输入图像为灰度图，将其转换为3通道RGB图像
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)通过 Transformer 对图像进行编码
        x = self.decoder(x, features)# 通过 DecoderCup 对特征进行解码
        logits = self.segmentation_head(x) # 通过 SegmentationHead 输出类别概率
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    # 'ViT-B_16': configs.get_b16_config(),
    'ViT-B_16': configs.get_r50_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}



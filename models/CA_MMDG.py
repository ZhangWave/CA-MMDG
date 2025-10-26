import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import sys
import numpy as np
from torch.autograd import Variable
import random
import os

from help.MSMF import MSMF
from pretrained_model import *
import torch.nn.functional as F  # 导入模块并重命名为 F

def l2_norm(input, axis=1):
    """
        L2 正则化函数，返回归一化后的张量。

        参数:
            input (Tensor): 输入张量
            axis (int): 归一化的维度

        返回:
            Tensor: 归一化后的张量
        """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Feature_Generator_MADDG(nn.Module):
    def __init__(self):
        """
                特征生成器初始化函数，定义了所有的卷积层、批量归一化层和 ReLU 激活函数。
                """
        super(Feature_Generator_MADDG, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv1_3 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(196)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv1_4 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(128)
        self.relu1_4 = nn.ReLU(inplace=True)
        self.maxpool1_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_5 = nn.BatchNorm2d(128)
        self.relu1_5 = nn.ReLU(inplace=True)
        self.conv1_6 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_6 = nn.BatchNorm2d(196)
        self.relu1_6 = nn.ReLU(inplace=True)
        self.conv1_7 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_7 = nn.BatchNorm2d(128)
        self.relu1_7 = nn.ReLU(inplace=True)
        self.maxpool1_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_8 = nn.BatchNorm2d(128)
        self.relu1_8 = nn.ReLU(inplace=True)
        self.conv1_9 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_9 = nn.BatchNorm2d(196)
        self.relu1_9 = nn.ReLU(inplace=True)
        self.conv1_10 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_10 = nn.BatchNorm2d(128)
        self.relu1_10 = nn.ReLU(inplace=True)
        self.maxpool1_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = self.conv1_1(out)
        out = self.bn1_1(out)
        out = self.relu1_1(out)
        out = self.conv1_2(out)
        out = self.bn1_2(out)
        out = self.relu1_2(out)
        out = self.conv1_3(out)
        out = self.bn1_3(out)
        out = self.relu1_3(out)
        out = self.conv1_4(out)
        out = self.bn1_4(out)
        out = self.relu1_4(out)
        pool_out1 = self.maxpool1_1(out)

        out = self.conv1_5(pool_out1)
        out = self.bn1_5(out)
        out = self.relu1_5(out)
        out = self.conv1_6(out)
        out = self.bn1_6(out)
        out = self.relu1_6(out)
        out = self.conv1_7(out)
        out = self.bn1_7(out)
        out = self.relu1_7(out)
        pool_out2 = self.maxpool1_2(out)

        out = self.conv1_8(pool_out2)
        out = self.bn1_8(out)
        out = self.relu1_8(out)
        out = self.conv1_9(out)
        out = self.bn1_9(out)
        out = self.relu1_9(out)
        out = self.conv1_10(out)
        out = self.bn1_10(out)
        out = self.relu1_10(out)
        pool_out3 = self.maxpool1_3(out)
        return pool_out3


class Feature_Embedder_MADDG(nn.Module):
    def __init__(self):
        """
特征嵌入器初始化函数，定义了所有的卷积层、池化层和全连接层。
  """
        super(Feature_Embedder_MADDG, self).__init__()
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottleneck_layer_1 = nn.Sequential(
            self.conv3_1,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.pool2_1,
            self.conv3_2,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self.pool2_2,
            self.conv3_3,
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)  # 修改为自适应平均池化
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, norm_flag):
        feature = self.bottleneck_layer_1(input)
        # print(f"Feature size before avg_pool: {feature.size()}")

        feature = self.avg_pool(feature)  # 将生成的特征和标准化标志传入特征嵌入器中

        # 打印特征图尺寸
        # print(f"Feature size after avg_pool: {feature.size()}")
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # change your path
    model_path = r'/root/code/SSDG-mm-r/pretrained_model/resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    # print(model)
    return model


class Feature_Generator_ResNet18(nn.Module):
    def __init__(self):
        """
        特征生成器初始化函数，使用预训练的ResNet-18模型并提取其部分层作为特征生成器。
        """
        super(Feature_Generator_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=True)
        # self.conv1x1 = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3

    def forward(self, input):
        # feature=self.conv1x1(input)
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        return feature


class Feature_Embedder_ResNet18(nn.Module):
    def __init__(self):
        """
                特征嵌入器初始化函数，使用ResNet-18模型的最后一层并添加额外的瓶颈层。
                """
        super(Feature_Embedder_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=False)
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, norm_flag):
        feature = self.layer4(input)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature


class Classifier(nn.Module):
    def __init__(self):
        """
        分类器初始化函数，定义一个线性层用于分类。
        """
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if (norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out


# 定义自定义的 GRL 类
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.iter_num = getattr(ctx, 'iter_num', 0) + 1
        ctx.alpha = 10
        ctx.low = 0.0
        ctx.high = 1.0
        ctx.max_iter = 4000
        return input

    @staticmethod
    def backward(ctx, grad_output):
        iter_num_tensor = torch.tensor(ctx.iter_num, dtype=torch.float32)
        coeff = 2.0 * (ctx.high - ctx.low) / (1.0 + torch.exp(-ctx.alpha * iter_num_tensor / ctx.max_iter)) - (
                    ctx.high - ctx.low) + ctx.low
        # coeff = torch.tensor(coeff, dtype=torch.float32)
        coeff = torch.tensor(coeff, dtype=torch.float32).clone().detach()

        return -coeff * grad_output



class MultiModalFeatureExtractor(nn.Module):
    def __init__(self):
        super(MultiModalFeatureExtractor, self).__init__()
        # RGB 特征提取器
        self.rgb_extractor = Feature_Generator_ResNet18()
        # Depth 特征提取器
        self.depth_extractor = Feature_Generator_ResNet18()
        # IR 特征提取器
        self.ir_extractor = Feature_Generator_ResNet18()

        # self.fusion_conv = nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=1)
        # 使用 MSMF 替代原来的 fusion_conv
        self.msmf = MSMF(channels=256)

    def forward(self, rgb_input, depth_input, ir_input):
        # rgb_input, depth_input, ir_input = feature[:, 3:6, :, :], feature[:, 0:3, :, :], feature[:, 6:, :, :]
        # 提取各模态特征
        rgb_feature = self.rgb_extractor(rgb_input)
        depth_feature = self.depth_extractor(depth_input)
        ir_feature = self.ir_extractor(ir_input)
        # fused_feature = torch.cat([ir_feature, depth_feature], dim=1)
        # 将深度和红外特征融合
        # fused_feature = self.msmf(F_b=depth_feature, F_a=ir_feature)
        # fused_feature = self.fusion_conv(fused_feature)


        return rgb_feature, depth_feature,ir_feature


class MultiModalFeatureEmbedder(nn.Module):
    def __init__(self):
        super(MultiModalFeatureEmbedder, self).__init__()
        # RGB 特征提取器
        self.rgb_embedder = Feature_Embedder_ResNet18()
        # Depth 特征提取器
        self.depth_embedder = Feature_Embedder_ResNet18()
        # IR 特征提取器
        self.ir_embedder = Feature_Embedder_ResNet18()
        # 特征融合模块（例如拼接 + 卷积）
        self.fusion_conv = nn.Conv2d(in_channels=256 * 3, out_channels=512, kernel_size=1)

    def forward(self, rgb_input, depth_input, ir_input):
        # rgb_input, depth_input, ir_input = feature[:, 3:6, :, :], feature[:, 0:3, :, :], feature[:, 6:, :, :]
        # 提取各模态特征
        rgb_feature = self.rgb_embedder(rgb_input)
        depth_feature = self.depth_embedder(depth_input)
        ir_feature = self.ir_embedder(ir_input)
        # print("ir_feature.shape",ir_feature.shape)
        # 拼接多模态特征
        fused_feature = torch.cat([rgb_feature, depth_feature, ir_feature], dim=1)
        # print("fused_feature.shape",fused_feature.shape)
        # 融合特征
        fused_feature = self.fusion_conv(fused_feature)
        return fused_feature


class FeatureDisentanglementWithFusion(nn.Module):
    def __init__(self):
        super(FeatureDisentanglementWithFusion, self).__init__()
        # 域共享特征提取
        self.domain_shared_fc = nn.Linear(512, 512)
        # 域特定特征提取
        self.domain_specific_fc = nn.Linear(512, 512)
        # 多模态特征融合模块（通道注意力）
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )

    def forward(self, fused_feature):
        # 展平特征：[batch_size, channels, height, width] -> [batch_size, channels * height * width]
        batch_size = fused_feature.size(0)
        fused_feature = fused_feature.view(batch_size, -1)  # 展平为二维张量
        # 解耦特征
        domain_shared = self.domain_shared_fc(fused_feature)
        domain_specific = self.domain_specific_fc(fused_feature)
        # 多模态特征融合
        attention_weights = self.attention(fused_feature)
        fused_output = attention_weights * domain_shared + (1 - attention_weights) * domain_specific
        return fused_output, domain_shared, domain_specific


class MultiTaskDiscriminator(nn.Module):
    def __init__(self):
        super(MultiTaskDiscriminator, self).__init__()
        # 共享特征提取层
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # 域分类器
        self.domain_classifier = nn.Linear(512, 3)  # 假设有 3 个域
        # 模态分类器
        self.modality_classifier = nn.Linear(512, 3)  # 假设有 3 种模态
        # 梯度反转层
        self.grl = GRL.apply

    def forward(self, feature):
        feature = self.grl(feature)
        shared_feature = self.shared_fc(feature)
        # 域分类
        domain_out = self.domain_classifier(shared_feature)
        # 模态分类
        modality_out = self.modality_classifier(shared_feature)
        return domain_out, modality_out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL.apply

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out

class CM_CPF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 多尺度特征提取（保持输出通道数相同）
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, stride=2 ** i, padding=1)
            for i in range(3)
        ])

        # 跨模态对比投影头
        self.proj = nn.Sequential(
            nn.Linear(channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # 对比损失权重
        self.tau = 0.1
        # 添加通道调整层
        self.channel_adjust = nn.Conv2d(3 * channels, channels, kernel_size=1)
    def forward(self, rgb, depth, ir):
        losses = []
        fused_features = []
        target_size = rgb.shape[-2:]  # 获取原始特征尺寸

        for conv in self.scale_convs:
            rgb_s = conv(rgb)
            depth_s = conv(depth)
            ir_s = conv(ir)

            # 上采样到原始尺寸
            rgb_s = F.interpolate(rgb_s, size=target_size, mode='bilinear', align_corners=False)
            depth_s = F.interpolate(depth_s, size=target_size, mode='bilinear', align_corners=False)
            ir_s = F.interpolate(ir_s, size=target_size, mode='bilinear', align_corners=False)

            # 对比损失计算（示例：RGB与Depth）
            z_rgb = self.proj(rgb_s.mean(dim=[2, 3]))
            z_depth = self.proj(depth_s.mean(dim=[2, 3]))
            sim = F.cosine_similarity(z_rgb, z_depth, dim=-1) / self.tau
            loss = -torch.log(torch.exp(sim) / (torch.exp(sim) + torch.exp(1 - sim)))
            losses.append(loss.mean())

            # 特征拼接
            fused = torch.cat([rgb_s, depth_s, ir_s], dim=1)
            fused = self.channel_adjust(fused)
            fused_features.append(fused)

        # 沿新维度堆叠并平均
        return torch.stack(fused_features, dim=1).mean(dim=1), sum(losses) / len(losses)


class DG_model(nn.Module):
    def __init__(self, model):
        super(DG_model, self).__init__()
        # 多流特征提取器
        self.feature_extractor = MultiModalFeatureExtractor()

        # self.fusion = MSMF(channels=256)  # 假设每个模态的特征通道数是 256
        # self.fusion = DG_MIN(channels=256)  # 假设每个模态的特征通道数是 256
        self.fusion = CM_CPF(channels=256)

        self.embedder = Feature_Embedder_ResNet18()
        # 分类器
        self.classifier = Classifier()

    def forward(self, input, norm_flag):
        rgb_input, depth_input, ir_input = input[:, 3:6, :, :], input[:, 0:3, :, :], input[:, 6:, :, :]

        # 提取多模态特征
        rgb_feature, depth , ir = self.feature_extractor(rgb_input, depth_input, ir_input)
        # 特征融合
        # 特征融合（解包融合特征和对比损失）
        fused_feature, contrast_loss = self.fusion(rgb_feature, depth, ir)  # 关键修改点
        # print("fused_feature.shape",fused_feature.shape)  # 应为 [B, C, H, W]
        feature = self.embedder(fused_feature, norm_flag)

        classifier_out = self.classifier(feature, norm_flag)
        return classifier_out, feature, contrast_loss



if __name__ == '__main__':
    x = Variable(torch.ones(5, 9, 256, 256))
    model = DG_model(model='resnet18')
    y, v = model(x, True)
    print(y.shape)
    print(y)
    print(v.shape)








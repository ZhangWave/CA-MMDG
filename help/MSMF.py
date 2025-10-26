import torch
import torch.nn as nn
import torch.nn.functional as F
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二层卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MSMF(nn.Module):
    def __init__(self, channels):
        """
        Master-Slave Modulation Fusion (MSMF) Module.

        Args:
            channels (int): Number of channels in the input feature maps.
        """
        super(MSMF, self).__init__()
        self.channels = channels

        # 1x1 convolutions for query, key, and value projections
        self.query_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # ResBlock (convolution + ReLU)
        self.res_block = ResBlock(channels,channels)

    def forward(self, F_b, F_a):
        """
        Forward pass of the MSMF module.

        Args:
            F_b (torch.Tensor): Baseline stream features, shape (B, C, H, W).
            F_a (torch.Tensor): Auxiliary contrastive stream features, shape (B, C, H, W).

        Returns:
            torch.Tensor: Fused features, shape (B, C, H, W).
        """
        B, C, H, W = F_b.shape

        # Generate query, key, and value
        Q_b = self.query_conv(F_b).view(B, C, -1)  # (B, C, H*W)
        K_a = self.key_conv(F_a).view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        V_a = self.value_conv(F_a).view(B, C, -1)  # (B, C, H*W)

        # Compute cross-modal attention weights
        attention_weights = torch.softmax(torch.bmm(K_a, Q_b), dim=-1)  # (B, H*W, H*W)

        # Modulate the auxiliary features
        M_a = torch.bmm(V_a, attention_weights)  # (B, C, H*W)
        M_a = M_a.view(B, C, H, W)  # Reshape to (B, C, H, W)

        # Add modulated features to baseline stream
        fused_features = F_b + M_a

        # Pass through ResBlock
        output = self.res_block(fused_features)

        # Skip connection
        output = output + fused_features

        return output


if __name__ == '__main__':
    # 定义输入特征
    B, C, H, W = 2, 64, 32, 32  # Batch size, channels, height, width
    F_b = torch.randn(B, C, H, W)  # Baseline stream features
    F_a = torch.randn(B, C, H, W)  # Auxiliary contrastive stream features

    # 初始化 MSMF 模块
    msmf = MSMF(channels=C)

    # 前向传播
    output = msmf(F_b, F_a)

    print("Input F_b shape:", F_b.shape)
    print("Input F_a shape:", F_a.shape)
    print("Output shape:", output.shape)


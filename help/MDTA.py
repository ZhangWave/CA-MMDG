import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(MDTA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Point-wise conv (1x1), depth-wise conv (3x3)
        def pw_dw_conv():
            return nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),  # point-wise
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)  # depth-wise
            )

        self.q_proj = pw_dw_conv()
        self.k_proj = pw_dw_conv()
        self.v_proj = pw_dw_conv()

        # Learnable scaling factor alpha
        self.scale = nn.Parameter(torch.ones(1))

        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape  # [B, C, H, W]

        # 1. 生成 Q, K, V，保留局部上下文
        Q = self.q_proj(x)  # [B, C, H, W]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. reshape 为多头形式： [B, num_heads, H*W, head_dim]
        def reshape_input(t):
            return t.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, num_heads, HW, head_dim]

        Q_ = reshape_input(Q)  # [B, heads, HW, d]
        K_ = reshape_input(K).permute(0, 1, 3, 2)  # [B, heads, d, HW]
        V_ = reshape_input(V)  # [B, heads, HW, d]

        # 3. attention map: QK^T → softmax → attention → V
        A = torch.matmul(Q_, K_) / self.scale  # [B, heads, HW, HW]
        A = F.softmax(A, dim=-1)
        out = torch.matmul(A, V_)  # [B, heads, HW, d]

        # 4. reshape 回原始通道维度
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        # 5. 残差连接 + 投影
        out = self.out_proj(out)
        return out + x  # residual connection

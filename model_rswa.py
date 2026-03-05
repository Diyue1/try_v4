import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

class WindowTiling(nn.Module):
    def forward(self, x_dwt):
        B, C4, H_half, W_half = x_dwt.shape
        C = C4 // 4
        LL, LH, HL, HH = torch.split(x_dwt, C, dim=1)
        combined = torch.stack([LL, LH, HL, HH], dim=2)
        combined = combined.view(B, C, 2, 2, H_half, W_half)
        combined = combined.permute(0, 1, 4, 2, 5, 3).contiguous()
        return combined.view(B, C, H_half * 2, W_half * 2)

class WindowRestore(nn.Module):
    def forward(self, x_tiled):
        B, C, H, W = x_tiled.shape
        H_half, W_half = H // 2, W // 2
        x = x_tiled.view(B, C, H_half, 2, W_half, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(B, 4 * C, H_half, W_half)

class ScaleDot(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return x * self.scale


def haar_dwt(x):
    B, C, H, W = x.shape
    if H % 2 != 0 or W % 2 != 0:
        x = F.pad(x, (0, W % 2, 0, H % 2))

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    LL = (x00 + x01 + x10 + x11) / 2
    LH = (x00 - x01 + x10 - x11) / 2
    HL = (x00 + x01 - x10 - x11) / 2
    HH = (x00 - x01 - x10 + x11) / 2

    return torch.cat([LL, LH, HL, HH], dim=1)  # (B, 4C, H/2, W/2)


def haar_idwt(x_dwt):
    B, C4, H_half, W_half = x_dwt.shape
    C = C4 // 4
    LL, LH, HL, HH = torch.split(x_dwt, C, dim=1)

    x00 = (LL + LH + HL + HH) / 2
    x01 = (LL - LH + HL - HH) / 2
    x10 = (LL + LH - HL - HH) / 2
    x11 = (LL - LH - HL + HH) / 2

    out = torch.zeros(B, C, H_half * 2, W_half * 2, device=x_dwt.device)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11

    return out


class GMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),  # [新增] 20% 的神经元随机失活
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.2)   # [新增]
        )

    def forward(self, x):
        return self.net(x)


# 将此代码块放入 model_rswa.py 中，替换原有的 RSWABlock 类

class RSWABlock(nn.Module):
    def __init__(self, dim, window_size, num_heads=4):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 论文提到的预处理: 1x1 conv + 3x3 depthwise conv [cite: 162]
        self.preprocess = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        )
        self.norm = LayerNorm2d(dim)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1)  # 对应公式(3)中的外部 conv

        # GMLP 分支 [cite: 160]
        self.gmlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        x = self.preprocess(x)
        x = self.norm(x)

        # 窗口切分
        pad_h = (self.ws - H % self.ws) % self.ws
        pad_w = (self.ws - W % self.ws) % self.ws
        x = F.pad(x, (0, pad_w, 0, pad_h))

        pH, pW = x.shape[2], x.shape[3]
        nH, nW = pH // self.ws, pW // self.ws

        # 展平窗口进行计算
        patches = x.unfold(2, self.ws, self.ws).unfold(3, self.ws, self.ws)
        patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, self.ws * self.ws, C)

        # LoAttention 分支 [cite: 166]
        qkv = self.qkv(patches).view(-1, self.ws * self.ws, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out_attn = (attn @ v).transpose(1, 2).reshape(-1, self.ws * self.ws, C)

        # GMLP 分支 [cite: 175]
        out_gmlp = self.gmlp(patches)

        # 融合逻辑对齐公式(3): Attention * GMLP
        out_fused = out_attn * out_gmlp

        # 还原空间维度
        out = out_fused.view(B, nH, nW, self.ws, self.ws, C).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, pH, pW)
        if pad_h > 0 or pad_w > 0: out = out[:, :, :H, :W]

        return self.proj(out) + shortcut

class ResNetClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        def conv3x3(in_planes, out_planes, stride=1):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super().__init__()
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = nn.BatchNorm2d(planes)
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

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class AIGCDetector(nn.Module):
    def __init__(self, lambda_fuse=0.4):
        super().__init__()
        self.lambda_fuse = lambda_fuse  # 论文最优值 0.4 [cite: 248]

        # DWT 分支
        self.tiling = WindowTiling()
        self.dwt_block = RSWABlock(96, window_size=4)  # 论文设置 b=4
        self.restore = WindowRestore()
        self.dwt_proj = nn.Conv2d(3, 96, 1)  # 映射到 embed_dim

        # FFT 分支
        self.fft_block = RSWABlock(96, window_size=8)  # 论文设置 b=8
        self.fft_proj = nn.Conv2d(3, 96, 1)
        self.fft_out_proj = nn.Conv2d(96, 3, 1)
        self.norm_fft = LayerNorm2d(96)

        self.classifier = ResNetClassifier(3, 2)  # 您代码中已有的分类器

    def forward(self, x):
        # 1. DWT Branch (Window Tiling Branch) [cite: 187]
        x_dwt = haar_dwt(x)  # (B, 12, H/2, W/2)
        feat_dwt = self.dwt_proj(F.interpolate(x, scale_factor=0.5))  # 基础特征
        # 接入平铺逻辑
        x_tiled = self.tiling(x_dwt)  # (B, 3, H, W)
        feat_tiled = self.dwt_block(self.dwt_proj(x_tiled))
        img_dwt = haar_idwt(self.restore(feat_tiled))

        # 2. FFT Branch (Phase Complement Branch)
        fft_x = torch.fft.fft2(x.float())
        amp, phase = torch.abs(fft_x), torch.angle(fft_x)

        feat_phase = self.fft_proj(phase)
        feat_phase = self.norm_fft(feat_phase)
        feat_phase = self.fft_block(feat_phase)

        new_phase = self.fft_out_proj(feat_phase)
        img_fft = torch.fft.ifft2(torch.polar(amp, new_phase)).real.to(x.dtype)

        # 3. Dual Branch Fusion [cite: 215]
        X_fused = (1 - self.lambda_fuse) * img_dwt + self.lambda_fuse * img_fft
        return self.classifier(X_fused)
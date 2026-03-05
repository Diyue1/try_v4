import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps) # [cite: 205]

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

class WindowTiling(nn.Module):
    """ 实现论文图3的 DWT 窗口平铺逻辑 [cite: 155, 193] """
    def forward(self, x_dwt):
        B, C4, H_half, W_half = x_dwt.shape
        C = C4 // 4
        LL, LH, HL, HH = torch.split(x_dwt, C, dim=1)
        combined = torch.stack([LL, LH, HL, HH], dim=2)
        combined = combined.view(B, C, 2, 2, H_half, W_half)
        combined = combined.permute(0, 1, 4, 2, 5, 3).contiguous()
        return combined.view(B, C, H_half * 2, W_half * 2)

class WindowRestore(nn.Module):
    """ 将平铺特征还原回四子带结构 [cite: 208] """
    def forward(self, x_tiled):
        B, C, H, W = x_tiled.shape
        H_half, W_half = H // 2, W // 2
        x = x_tiled.view(B, C, H_half, 2, W_half, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(B, 4 * C, H_half, W_half)

def haar_dwt(x):
    B, C, H, W = x.shape
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    LL = (x00 + x01 + x10 + x11) / 2
    LH = (x00 - x01 + x10 - x11) / 2
    HL = (x00 + x01 - x10 - x11) / 2
    HH = (x00 - x01 - x10 + x11) / 2
    return torch.cat([LL, LH, HL, HH], dim=1) # [cite: 191]

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

class RSWABlock(nn.Module):
    def __init__(self, dim, window_size, num_heads=4):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 预处理：1x1 conv + 3x3 depthwise conv [cite: 162]
        self.preprocess = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        )
        self.norm = LayerNorm2d(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.gmlp = nn.Sequential( # [cite: 175, 179]
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.proj = nn.Conv2d(dim, dim, 1) # [cite: 184]

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = self.preprocess(x)
        x = self.norm(x)

        pad_h = (self.ws - H % self.ws) % self.ws
        pad_w = (self.ws - W % self.ws) % self.ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        pH, pW = x.shape[2], x.shape[3]
        nH, nW = pH // self.ws, pW // self.ws

        patches = x.unfold(2, self.ws, self.ws).unfold(3, self.ws, self.ws)
        patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, self.ws * self.ws, C)

        # LoAttention [cite: 172]
        qkv = self.qkv(patches).view(-1, self.ws * self.ws, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out_attn = (attn @ v).transpose(1, 2).reshape(-1, self.ws * self.ws, C)

        # GMLP & Fusion (公式 3) [cite: 184]
        out_fused = out_attn * self.gmlp(patches)

        out = out_fused.view(B, nH, nW, self.ws, self.ws, C).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, pH, pW)
        if pad_h > 0 or pad_w > 0: out = out[:, :, :H, :W]
        return self.proj(out) + shortcut

class ResNetClassifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 标准 ResNet 结构逻辑 [cite: 240]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 对应复现论文的 BCELoss，输出维度改为 1 
        self.fc = nn.Linear(512, 1)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [self._BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(self._BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    class _BasicBlock(nn.Module):
        def __init__(self, in_p, p, s=1, d=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_p, p, 3, s, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(p)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(p, p, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(p)
            self.downsample = d
        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample: identity = self.downsample(x)
            return self.relu(out + identity)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))

class AIGCDetector(nn.Module):
    def __init__(self, embed_dim=96, lambda_fuse=0.4):
        super().__init__()
        self.lambda_fuse = lambda_fuse # 
        
        # DWT Branch (b=4) [cite: 206]
        self.tiling = WindowTiling()
        self.dwt_proj = nn.Conv2d(3, embed_dim, 1)
        self.dwt_block = RSWABlock(embed_dim, window_size=4)
        self.restore = WindowRestore()
        self.idwt_proj = nn.Conv2d(embed_dim * 4, 12, 1)

        # FFT Branch (b=8, Phase-only) [cite: 211]
        self.fft_proj = nn.Conv2d(3, embed_dim, 1)
        self.norm_fft = LayerNorm2d(embed_dim)
        self.fft_block = RSWABlock(embed_dim, window_size=8)
        self.fft_out_proj = nn.Conv2d(embed_dim, 3, 1)

        self.classifier = ResNetClassifier(in_channels=3)

    def forward(self, x):
        # DWT Branch
        x_tiled = self.tiling(haar_dwt(x))
        feat_dwt = self.dwt_block(self.dwt_proj(x_tiled))
        img_dwt = haar_idwt(self.idwt_proj(self.restore(feat_dwt)))

        # FFT Branch (Phase Part) [cite: 209, 212]
        fft_x = torch.fft.fft2(x.float())
        amp, phase = torch.abs(fft_x), torch.angle(fft_x)
        feat_phase = self.fft_block(self.norm_fft(self.fft_proj(phase)))
        new_phase = self.fft_out_proj(feat_phase)
        img_fft = torch.fft.ifft2(torch.polar(amp, new_phase.float())).real.to(x.dtype)

        # Fusion & Classification (公式 5) [cite: 215]
        X_fused = (1 - self.lambda_fuse) * img_dwt + self.lambda_fuse * img_fft
        # BCELoss 需要 Sigmoid 后的单维概率 
        return self.classifier(X_fused).view(-1)


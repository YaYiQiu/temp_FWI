import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .util import wavelet as wavelet_utils  
from NLP import nlp_model

class RedundantWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, level=3, wavelet_type='db4', use_bias=True):
        super(RedundantWTConv2d, self).__init__()
        assert in_channels == out_channels, "要求输入输出通道数相同"

        self.nlp_model = nlp_model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.kernel_size = kernel_size

        # 普通卷积残差分支
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  padding=kernel_size // 2, bias=use_bias)

        # 每层小波融合后卷积器
        self.band_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=use_bias)
            for _ in range(level)
        ])

        # 每层缩放参数
        self.band_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(level)
        ])

        # 预加载固定小波滤波器
        dec_filter, rec_filter = wavelet_utils.create_wavelet_filter(
            wavelet_type, in_channels, in_channels, torch.float)
        self.register_buffer("dec_filter", dec_filter)
        self.register_buffer("rec_filter", rec_filter)
        self.wt = partial(wavelet_utils.wavelet_transform, filters=self.dec_filter)
        self.iwt = partial(wavelet_utils.inverse_wavelet_transform, filters=self.rec_filter)

    def forward(self, x):
        """
        x: [B,C,H,W]
        fusion_weights: dict
            {
                'level0': {'lh': w_lh, 'hl': w_hl, 'hh': w_hh},
                'level1': {...},
                'level2': {...}
            }
        其中 w_lh 等权重张量形状可为 (B,1,1,1) 或 (B,C,1,1)
        """
        ##fusion_weights = self.nlp_model.get_fusion_weights(x)
        fusion_weights = {
            f'level{l}': {
                'll': 0.8,
                'lh': 1.2,
                'hl': 1.2,
                'hh': 1.5
            } for l in range(self.level)
        }
        # fusion_weights = {
        #     'level0': {'ll': 0.8, 'lh': 1.2, 'hl': 1.2, 'hh': 1.5},
        #     'level1': {'ll': 0.5, 'lh': 1.1, 'hl': 1.1, 'hh': 1.3},
        #     'level2': {'ll': 0.3, 'lh': 1.0, 'hl': 1.0, 'hh': 1.0}
        # }
        residual = self.res_conv(x)  # 普通卷积残差

        current = x
        outputs = []

        for l in range(self.level):
            wt = self.wt(current)  # (B,C,4,H/2,W/2)
            ll, lh, hl, hh = wt[:, :, 0], wt[:, :, 1], wt[:, :, 2], wt[:, :, 3]

            fw = fusion_weights.get(f'level{l}', {})
            w_lh = fw.get('lh', 1.0)
            w_hl = fw.get('hl', 1.0)
            w_hh = fw.get('hh', 1.0)

            # 权重广播匹配
            if isinstance(w_lh, (int, float)):
                w_lh = torch.tensor(w_lh, device=x.device).view(1, 1, 1, 1)
            if isinstance(w_hl, (int, float)):
                w_hl = torch.tensor(w_hl, device=x.device).view(1, 1, 1, 1)
            if isinstance(w_hh, (int, float)):
                w_hh = torch.tensor(w_hh, device=x.device).view(1, 1, 1, 1)

            fused_band = lh * w_lh + hl * w_hl + hh * w_hh  # 融合频带

            # 卷积 + 缩放
            band_conv = self.band_convs[l](fused_band)
            band_conv = self.band_scales[l] * band_conv

            # 组合4个子带用于反变换 (ll + 融合频带 + 0 + 0)
            dummy = torch.zeros_like(ll)
            combined = torch.stack([ll, fused_band, dummy, dummy], dim=2)  # (B,C,4,H/2,W/2)
            current = self.iwt(combined)  # 反小波重建，恢复大小

            # 对齐 residual 大小
            if current.shape[2:] != residual.shape[2:]:
                current = F.interpolate(current, size=residual.shape[2:], mode="bilinear", align_corners=False)

            outputs.append(current)

        out = residual + sum(outputs)
        return F.relu(out)

import pywt
import pywt.data
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    """
    冗余小波变换
    输入: x [B, C, H, W]
    输出: [B, C, 4, H_out, W_out]
    """
    b, c, _, _ = x.shape
    pad = (filters.shape[2] // 2, filters.shape[3] // 2)
    
    # stride=2 降采样
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)

    # 自动获取卷积后的 H, W
    H_out, W_out = x.shape[2], x.shape[3]

    # reshape 时用实际 H_out, W_out
    x = x.reshape(b, c, 4, H_out, W_out)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2, filters.shape[3] // 2)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x 

def wavelet_decompose(x, wavelet='db4', level=3):
    filters, _ = create_wavelet_filter(wavelet, x.size(1), x.size(1), x.dtype)
    coeffs = []
    current = x

    for _ in range(level):
        x_wt = wavelet_transform(current, filters)  # shape: [B, C, 4, H//2, W//2]
        # 提取子带
        bands = {
            'LL': x_wt[:,:,0,:,:],
            'LH': x_wt[:,:,1,:,:],
            'HL': x_wt[:,:,2,:,:],
            'HH': x_wt[:,:,3,:,:],
        }
        coeffs.append(bands)
        current = bands['LL']  # 继续对LL分解

    return coeffs

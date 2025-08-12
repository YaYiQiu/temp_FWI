import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.show import *

from wtconv import RedundantWTConv2d
from wtconv.util.wavelet import wavelet_decompose 

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, structural_similarity

NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.PReLU())
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                                     padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.PReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class PUW_FWI(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512):
        super(PUW_FWI, self).__init__()

        # ------------------------- Encoder -------------------------
        # Stage 1
        self.enc1_wt = RedundantWTConv2d(5, 5, level=3)
        self.enc1_conv1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.enc1_conv2 = ConvBlock(dim1, dim1, kernel_size=(3, 1), padding=(1, 0))

        # Stage 2
        self.enc2_wt = RedundantWTConv2d(dim1, dim1, level=3)
        self.enc2_conv1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.enc2_conv2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))

        # Stage 3
        self.enc3_wt = RedundantWTConv2d(dim2, dim2, level=3)
        self.enc3_conv1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.enc3_conv2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))

        # ------------------------- Bottleneck -------------------------
        self.bottleneck_wt = RedundantWTConv2d(dim3, dim3, level=3)
        self.bottleneck_conv1 = ConvBlock(dim3, dim4, kernel_size=(3, 3), padding=1)
        self.bottleneck_conv2 = ConvBlock(dim4, dim5, kernel_size=(3, 3), padding=1)  # 扩展到 dim5

        # ------------------------- Decoder -------------------------
        # 顶层先把 dim5 -> dim4
        self.dec5_upsample = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.dec5_conv = ConvBlock(dim4, dim4)

        # Stage -> dim3
        self.dec3_upsample = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.dec3_wt = RedundantWTConv2d(dim3, dim3, level=3)
        self.dec3_conv = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))

        # Stage -> dim2
        self.dec2_upsample = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.dec2_wt = RedundantWTConv2d(dim2, dim2, level=3)
        self.dec2_conv = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))

        # Stage -> dim1
        self.dec1_upsample = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.dec1_wt = RedundantWTConv2d(dim1, dim1, level=3)
        self.dec1_conv = ConvBlock(dim1, dim1, kernel_size=(3, 1), padding=(1, 0))

        # 输出层
        self.out_layer = ConvBlock_Tanh(dim1, 1)

        # ------------------------- Skip-channel 投影（1x1 conv） -------------------------
        # 为保证跳跃连接时 channel 可相加，准备必要的 1x1 投影：
        # enc3 (dim3=128) -> dec5 target channels dim4=256
        self.skip3_proj = nn.Conv2d(dim3, dim4, kernel_size=1)  # 将 x3 投影到 y5 的通道数
        # enc2 (dim2=64) -> dec3 target channels dim3=128
        self.skip2_proj = nn.Conv2d(dim2, dim3, kernel_size=1)
        # enc1 (dim1=32) -> dec2 target channels dim2=64
        self.skip1_proj = nn.Conv2d(dim1, dim2, kernel_size=1)

    # 类方法：把 src 的空间尺寸（H,W）对齐到 tgt 的尺寸；不改变 channel
    def _align_feature(self, src, tgt):
        if src.shape[2:] == tgt.shape[2:]:
            return src
        # 双线性插值，align_corners=False 更稳定
        return F.interpolate(src, size=tgt.shape[2:], mode='bilinear', align_corners=False)

    # 类方法：先对齐空间，再投影通道到目标的通道数（如果需要）
    def _align_and_proj(self, src, tgt, proj_conv: nn.Conv2d):
        """
        src: 待对齐张量 (B, C_src, H_src, W_src)
        tgt: 目标张量 (B, C_tgt, H_tgt, W_tgt)
        proj_conv: 1x1 conv 用于将 C_src -> C_tgt（如果 C_src != C_tgt）
        返回: src' (B, C_tgt, H_tgt, W_tgt)
        """
        src_al = self._align_feature(src, tgt)
        if src_al.shape[1] != tgt.shape[1]:
            # 使用提供的 1x1 conv 投影通道
            src_al = proj_conv(src_al)
        return src_al

    def forward(self, x, **kwargs):
        # ------------------------- Encoder -------------------------
        x1 = self.enc1_wt(x)                      # 冗余小波卷积（level=3）
        x1 = self.enc1_conv1(x1)                  # 卷积1
        x1 = self.enc1_conv2(x1)                  # 卷积2

        x2 = self.enc2_wt(x1)                     # 冗余小波卷积（level=3）
        x2 = self.enc2_conv1(x2)                  # 卷积1
        x2 = self.enc2_conv2(x2)                  # 卷积2

        x3 = self.enc3_wt(x2)                     # 冗余小波卷积（level=3）
        x3 = self.enc3_conv1(x3)                  # 卷积1
        x3 = self.enc3_conv2(x3)                  # 卷积2

        # ------------------------- Bottleneck -------------------------
        xb = self.bottleneck_wt(x3)               # 冗余小波卷积
        xb = self.bottleneck_conv1(xb)            # 卷积，dim3->dim4
        xb = self.bottleneck_conv2(xb)            # 卷积，dim4->dim5

        # ------------------------- Decoder -------------------------
        # 顶层： dim5 -> dim4
        y5 = self.dec5_upsample(xb)
        # 外部 skip3（可选）: 期望结构与 x3 相同通道，若不同则使用投影
        skip3 = kwargs.get('skip_stage3', None)

        # 对齐 x3 到 y5：先空间对齐，再通道投影到 y5 的 channels（使用 skip3_proj）
        x3_aligned = self._align_and_proj(x3, y5, self.skip3_proj)

        if skip3 is not None:
            # 假定 skip3 的语义通道类似 x3；先空间对齐，然后如有必要投影
            skip3_aligned = self._align_and_proj(skip3, y5, self.skip3_proj)
            y5 = y5 + x3_aligned + skip3_aligned
        else:
            y5 = y5 + x3_aligned

        y5 = self.dec5_conv(y5)

        # Stage -> dim3
        y3 = self.dec3_upsample(y5)

        skip2 = kwargs.get('skip_stage2', None)
        x2_aligned = self._align_and_proj(x2, y3, self.skip2_proj)
        if skip2 is not None:
            skip2_aligned = self._align_and_proj(skip2, y3, self.skip2_proj)
            y3 = y3 + x2_aligned + skip2_aligned
        else:
            y3 = y3 + x2_aligned

        y3 = self.dec3_wt(y3)
        y3 = self.dec3_conv(y3)

        # Stage -> dim2
        y2 = self.dec2_upsample(y3)

        skip1 = kwargs.get('skip_stage1', None)
        x1_aligned = self._align_and_proj(x1, y2, self.skip1_proj)
        if skip1 is not None:
            skip1_aligned = self._align_and_proj(skip1, y2, self.skip1_proj)
            y2 = y2 + x1_aligned + skip1_aligned
        else:
            y2 = y2 + x1_aligned

        y2 = self.dec2_wt(y2)
        y2 = self.dec2_conv(y2)

        # Stage -> dim1
        y1 = self.dec1_upsample(y2)
        y1 = self.dec1_wt(y1)
        y1 = self.dec1_conv(y1)

        # 输出层
        out = self.out_layer(y1)
        return out


def wavelet_loss(pred, target, wavelet='db4', level=3, lambda_s=1.0, lambda_w=0.7, lambda_e=0.3):
    """
    综合空间域 + 小波域结构误差 + 小波能量误差的多任务损失函数

    参数:
        pred: 模型输出张量，形状 [B, 1, H, W]
        target: Ground Truth 张量，形状 [B, 1, H, W]
        wavelet: 小波类型 (如 'db4')
        level: 小波分解层数（默认3）
        lambda_s: 空间域损失权重
        lambda_w: 小波结构损失权重
        lambda_e: 能量对齐损失权重
    """
    # 1. 空间域主损失
    spatial_loss = F.mse_loss(pred, target)

    # 2. 小波域结构损失和能量损失（只对最后一层高频频带）
    pred_coeffs = wavelet_decompose(pred, wavelet=wavelet, level=level)
    target_coeffs = wavelet_decompose(target, wavelet=wavelet, level=level)

    wavelet_mse = 0.0
    wavelet_energy = 0.0

    for b in range(len(pred_coeffs)):
        pred_band = pred_coeffs[b][level - 1]
        target_band = target_coeffs[b][level - 1]

        for band in ['LH', 'HL', 'HH']:
            p = pred_band[band]
            t = target_band[band]

            wavelet_mse += F.mse_loss(p, t)
            wavelet_energy += (p.pow(2).mean() - t.pow(2).mean()).abs()

    total_loss = (
        lambda_s * spatial_loss +
        lambda_w * wavelet_mse +
        lambda_e * wavelet_energy
    )

    return total_loss




if __name__ == '__main__':
    # 1. 创建模拟输入数据
    input = torch.rand((1, 5, 1000, 70))  # batch_size=1更合理
    
    # 2. 创建对应的真实速度模型（模拟数据）
    true_model = np.zeros((1000, 70))
    true_model[300:700, 20:50] = np.linspace(0.1, 0.8, 400)[:, np.newaxis]  # 渐变速度结构
    true_model += np.random.normal(0, 0.02, size=true_model.shape)  # 添加噪声
    
    # 3. 初始化模型
    model = PUW_FWI()
    
    # 4. 模型推理
    with torch.no_grad():
        output = model(input)
        
        # 处理不同输出类型
        if isinstance(output, (list, tuple)):
            print(f"模型返回{len(output)}个输出，使用第一个作为速度模型")
            velocity_model = output[0][0, 0].numpy()  # 取第一个输出的第一个样本第一个通道
        else:
            velocity_model = output[0, 0].numpy()  # 单输出情况
    
    # 5. 尺寸匹配检查
    if true_model.shape != velocity_model.shape:
        from skimage.transform import resize
        velocity_model = resize(velocity_model, true_model.shape, anti_aliasing=True)
        print(f"调整预测模型尺寸：{velocity_model.shape} → {true_model.shape}")
    
    # 6. 计算评估指标
    mse = mean_squared_error(true_model, velocity_model)
    ssim = structural_similarity(true_model, velocity_model, 
                               data_range=true_model.max()-true_model.min())
    snr = 10 * np.log10(np.sum(true_model**2) / np.sum((true_model - velocity_model)**2))
    
    # 7. 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 真实模型
    plt.subplot(1, 3, 1)
    plt.imshow(true_model, cmap='seismic', aspect='auto', vmin=0, vmax=0.8)
    plt.colorbar(label='Velocity (km/s)')
    plt.title("True Velocity Model")
    
    # 预测模型
    plt.subplot(1, 3, 2)
    plt.imshow(velocity_model, cmap='seismic', aspect='auto', vmin=0, vmax=0.8)
    plt.colorbar(label='Velocity (km/s)')
    plt.title(f"Predicted\nMSE={mse:.4f}")
    
    # 残差图
    plt.subplot(1, 3, 3)
    residual = true_model - velocity_model
    plt.imshow(residual, cmap='coolwarm', aspect='auto', vmin=-0.3, vmax=0.3)
    plt.colorbar(label='Residual (km/s)')
    plt.title(f"Residual\nSSIM={ssim:.4f}")
    
    plt.tight_layout()
    plt.savefig('velocity_comparison.png', dpi=300)
    plt.show()
    
    # 8. 输出评估报告
    report = f"""
=== 评估结果 ===
输入尺寸: 5×1000×70
输出尺寸: {velocity_model.shape}
MSE (均方误差): {mse:.5f} (越小越好)
SSIM (结构相似性): {ssim:.4f} (1.0为完美匹配)
SNR (信噪比): {snr:.2f} dB (越大越好)
真实速度范围: {true_model.min():.2f}~{true_model.max():.2f} km/s
预测速度范围: {velocity_model.min():.2f}~{velocity_model.max():.2f} km/s
"""
    print(report)
    with open("evaluation_report.txt", "w") as f:
        f.write(report)

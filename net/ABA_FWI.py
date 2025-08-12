# -*- coding: utf-8 -*-
"""
Created on 2024/7/16 20:11

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.show import *
from wtconv import *

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


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        y = self.fc(avg_pool) + self.fc(max_pool)
        y = torch.sigmoid(y)
        return x * y


class SpatialAttention1(nn.Module):
    def __init__(self):
        super(SpatialAttention1, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

    def forward(self, x):

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_pool, max_pool], dim=1)
        y = self.conv(y)
        y = torch.sigmoid(y)
        # import matplotlib.pyplot as plt
        # plt.imshow(x.detach().cpu().numpy()[102][15][:][:])
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_pool, max_pool], dim=1)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return x * y

class CBAMModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMModule, self).__init__()
        # self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        #  x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ABA_FWI(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(ABA_FWI, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(9, 6))
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8_1 = ConvBlock(dim4, dim5, stride=2)
        self.convblock8_2 = ConvBlock(dim5, dim5)

        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

        self.CBAM_model1 = CBAMModule(256)
        self.CBAM_model2 = CBAMModule(128)
        self.CBAM_model3 = CBAMModule(64)
        self.CBAM_model4 = CBAMModule(32)

        self.change_channel0 = nn.Conv2d(256, 128, kernel_size=1)
        self.change_channel1 = nn.Conv2d(512, 256, kernel_size=1)
        self.change_channel2 = nn.Conv2d(256, 128, kernel_size=1)

        self.wt0 = WTConv2d(128,128)
        self.wt1 = WTConv2d(128,128)
        self.wt2 = WTConv2d(256,256)

    def forward(self, x):
        # Encoder Part
        x0 = self.convblock1(x)  # (None, 32, 500, 70)
        x0 = self.convblock2_1(x0)  # (None, 64, 250, 70)
        x0 = self.convblock2_2(x0)  # (None, 64, 250, 70)
        x0 = self.convblock3_1(x0)  # (None, 64, 125, 70)
        x0 = self.convblock3_2(x0)  # (None, 64, 125, 70)

        x1 = self.convblock4_1(x0)  # (None, 128, 63, 70)
        x2 = self.convblock4_2(x1)  # (None, 128, 63, 70)
        x2_wt = self.wt0(x2)

        x3 = self.convblock5_1(x2_wt)  # (None, 128, 40, 40)
        x4 = self.convblock5_2(x3)  # (None, 128, 40, 40)
        x4_wt = self.wt1(x4)

        x5 = self.convblock6_1(x4_wt)  # (None, 256, 20, 20)
        x6 = self.convblock6_2(x5)  # (None, 256, 20, 20)
        x6_wt = self.wt2(x6)

        x7 = self.convblock7_1(x6_wt)  # (None, 256, 10, 10)
        x8 = self.convblock7_2(x7)  # (None, 256, 10, 10)

        x9 = self.convblock8_1(x8)  # (None, 512, 5, 5)
        x10 = self.convblock8_2(x9)  # (None, 512, 5, 5)

        # Decoder Part Vmodel
        y0 = self.deconv2_1(x10)  # (None, 256, 10, 10)
        y0_concat = torch.cat((x8, y0), dim=1)
        y0_concat = self.change_channel1(y0_concat)
        y1 = self.deconv2_2(y0_concat)  # (None, 256, 10, 10)
        y1_ca = self.CBAM_model1(y1)

        y2 = self.deconv3_1(y1_ca)  # (None, 128, 20, 20)
        x6_wt = self.change_channel0(x6_wt)
        y2_concat = torch.cat((x6_wt, y2), dim=1)
        y2_concat = self.change_channel2(y2_concat)
        y3 = self.deconv3_2(y2_concat)  # (None, 128, 20, 20)
        y3_ca = self.CBAM_model2(y3)

        y4 = self.deconv4_1(y3_ca)  # (None, 64, 40, 40)
        y5 = self.deconv4_2(y4)  # (None, 64, 40, 40)
        y5_ca = self.CBAM_model3(y5)

        y6 = self.deconv5_1(y5_ca)  # (None, 32, 80, 80)
        y7 = self.deconv5_2(y6)  # (None, 32, 80, 80)
        y7_ca = self.CBAM_model4(y7)

        # pain_openfwi_velocity_model(y7_ca[0,0,:,:].cpu().detach().numpy())
        y8 = F.pad(y7_ca, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        # pain_openfwi_velocity_model(y8[0,0,:,:].cpu().detach().numpy())
        y9 = self.deconv6(y8)  # (None, 1, 70, 70)

        return y9


class ABA_Loss(nn.Module):
    """
    The ablation experiment.
    Add skip connections into InversionNet.
    """
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(ABA_Loss, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(9, 6))
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8_1 = ConvBlock(dim4, dim5, stride=2)
        self.convblock8_2 = ConvBlock(dim5, dim5)

        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

        self.change_channel0 = nn.Conv2d(256, 128, kernel_size=1)
        self.change_channel1 = nn.Conv2d(512, 256, kernel_size=1)
        self.change_channel2 = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x):
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)
        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        x = self.convblock5_1(x)  # (None, 128, 40, 40)
        x = self.convblock5_2(x)  # (None, 128, 40, 40)

        x = self.convblock6_1(x)  # (None, 256, 20, 20)
        x = self.convblock6_2(x)  # (None, 256, 20, 20)

        x1 = x  # (None, 64, 20, 20)

        x = self.convblock7_1(x)  # (None, 256, 10, 10)
        x = self.convblock7_2(x)  # (None, 256, 10, 10)

        x2 = x  # (None, 64, 20, 20)

        x = self.convblock8_1(x)  # (None, 512, 5, 5)
        x = self.convblock8_2(x)  # (None, 512, 5, 5)

        # Decoder Part Vmodel
        y = self.deconv2_1(x)  # (None, 256, 10, 10)
        y_concat = torch.cat((x2, y), dim=1)
        y_concat = self.change_channel1(y_concat)
        y1 = self.deconv2_2(y_concat)  # (None, 256, 10, 10)

        y2 = self.deconv3_1(y1)  # (None, 128, 20, 20)
        x1 = self.change_channel0(x1)
        y2_concat = torch.cat((x1, y2), dim=1)
        y2_concat = self.change_channel2(y2_concat)
        y3 = self.deconv3_2(y2_concat)  # (None, 128, 20, 20)

        y4 = self.deconv4_1(y3)  # (None, 64, 40, 40)
        y5 = self.deconv4_2(y4)  # (None, 64, 40, 40)

        y6 = self.deconv5_1(y5)  # (None, 32, 80, 80)
        y7 = self.deconv5_2(y6)  # (None, 32, 80, 80)

        y8 = F.pad(y7, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        # pain_openfwi_velocity_model(y8[0,0,:,:].cpu().detach().numpy())
        y9 = self.deconv6(y8)  # (None, 1, 70, 70)

        return y9



if __name__ == '__main__':
    # 1. 创建模拟输入数据
    input = torch.rand((1, 5, 1000, 70))  # batch_size=1更合理
    
    # 2. 创建对应的真实速度模型（模拟数据）
    true_model = np.zeros((1000, 70))
    true_model[300:700, 20:50] = np.linspace(0.1, 0.8, 400)[:, np.newaxis]  # 渐变速度结构
    true_model += np.random.normal(0, 0.02, size=true_model.shape)  # 添加噪声
    
    # 3. 初始化模型
    model = ABA_FWI()
    
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

#输出结果
# ok
# tensor([[[[-0.4499,  0.1608, -0.7317,  ..., -0.7732, -0.1184, -0.2875],
#           [-0.7112,  0.1335, -0.0652,  ..., -0.2937, -0.5299, -0.0033],
#           [-0.1369,  0.3009, -0.7553,  ...,  0.0086, -0.7925, -0.4308],
#           ...,
#           [ 0.2811, -0.1191,  0.0886,  ..., -0.1697,  0.0917, -0.8443],
#           [-0.5533, -0.2984,  0.1013,  ...,  0.8234, -0.9298, -0.5550],
#           [-0.5530,  0.3660,  0.6752,  ...,  0.5136,  0.2138, -0.7413]]],


#         [[[-0.0077, -0.5390, -0.8514,  ..., -0.5271, -0.2585, -0.4148],
#           [-0.7928, -0.2524,  0.1223,  ..., -0.1394, -0.6481,  0.2783],
#           [-0.6951, -0.0964, -0.7179,  ..., -0.6486, -0.3692, -0.0537],
#           ...,
#           [ 0.4152, -0.3278, -0.2188,  ...,  0.3786, -0.6370,  0.5394],
#           [-0.1156, -0.6402, -0.0853,  ...,  0.9249, -0.7093, -0.9176],
#           [ 0.1183,  0.6346,  0.7463,  ..., -0.2066,  0.2650, -0.7479]]],


#         [[[-0.1463, -0.6346, -0.9114,  ..., -0.4422, -0.1666, -0.0626],
#           [-0.6145,  0.2934, -0.2773,  ..., -0.4501, -0.6931, -0.5285],
#           [-0.5866,  0.1642, -0.2002,  ...,  0.0570, -0.1030,  0.1348],
#           ...,
#           [ 0.4073, -0.6446,  0.0602,  ..., -0.7751,  0.7414, -0.7204],
#           [ 0.5492, -0.1032,  0.3908,  ...,  0.5074, -0.2969, -0.4328],
#           [ 0.0076,  0.7623,  0.4250,  ...,  0.7103,  0.7413, -0.8444]]],


#         [[[-0.0918, -0.0862, -0.5080,  ..., -0.3707, -0.3465, -0.1359],
#           [-0.6436, -0.1313,  0.3137,  ..., -0.8360, -0.3493, -0.5383],
#           [ 0.0581,  0.1351, -0.6728,  ..., -0.5648,  0.4902, -0.9163],
#           ...,
#           [ 0.2827,  0.0721, -0.1995,  ...,  0.6408, -0.2777, -0.5241],
#           [-0.2722, -0.6915,  0.0092,  ...,  0.8666,  0.8094, -0.0379],
#           [-0.3764,  0.1549,  0.4050,  ..., -0.3060,  0.3417, -0.3916]]],


#         [[[-0.2003, -0.3082, -0.6047,  ..., -0.9070,  0.5695, -0.2850],
#           [-0.5456, -0.1621, -0.6713,  ..., -0.2782, -0.7842,  0.4711],
#           [-0.4250,  0.0198, -0.1564,  ..., -0.0160, -0.8628, -0.2338],
#           ...,
#           [ 0.3725, -0.7950, -0.0024,  ..., -0.6024,  0.8336, -0.2369],
#           [-0.4247, -0.4157, -0.6985,  ...,  0.8195, -0.2983, -0.6517],
#           [-0.2209,  0.3882, -0.3507,  ...,  0.5495, -0.3076, -0.7898]]]],
#        grad_fn=<TanhBackward0>)
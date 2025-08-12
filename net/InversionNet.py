import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import mean_squared_error, structural_similarity

NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
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
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class InversionNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x):
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)
        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)
        x = self.convblock7_1(x)  # (None, 256, 8, 9)
        x = self.convblock7_2(x)  # (None, 256, 8, 9)
        x = self.convblock8(x)  # (None, 512, 1, 1)

        # Decoder Part Vmodel
        x = self.deconv1_1(x)  # (None, 512, 5, 5)
        x = self.deconv1_2(x)  # (None, 512, 5, 5)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        x = self.deconv3_1(x)  # (None, 128, 20, 20)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        x = self.deconv4_1(x)  # (None, 64, 40, 40)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        x = self.deconv5_1(x)  # (None, 32, 80, 80)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x)  # (None, 1, 70, 70)

        return x


class Discriminator(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, **kwargs):
        super(Discriminator, self).__init__()
        self.convblock1_1 = ConvBlock(1, dim1, stride=2)
        self.convblock1_2 = ConvBlock(dim1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = ConvBlock(dim2, dim2)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5 = ConvBlock(dim4, 1, kernel_size=5, padding=0)

    def forward(self, x):
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        return x


if __name__ == '__main__':
    # 1. 创建输入数据
    input = torch.zeros((1, 5, 1000, 70))  # 改为batch_size=1以便简化
    input[:, :, 300:700, 20:50] = 1.0  # 模拟水平反射层
    
    # 2. 创建对应的真实速度模型（必须与输出尺寸相同）
    true_model = np.zeros((1000, 70))  # 确保这个尺寸与模型输出一致
    true_model[300:700, 20:50] = 0.8
    
    # 3. 初始化模型
    model = InversionNet()
    try:
        model.load_state_dict(torch.load('model_weights.pth'))
    except:
        print("警告：未加载预训练权重，使用随机初始化")
    model.eval()
    
    # 4. 推理
    with torch.no_grad():
        output_tensor = model(input)
    
    # 5. 数据处理
    output = output_tensor.detach().cpu().numpy()
    velocity_model = output[0, 0, :, :]  # 假设输出是[batch, channel, height, width]
    
    # 6. 尺寸验证（关键修复步骤）
    if true_model.shape != velocity_model.shape:
        # 自动调整真实模型尺寸（根据实际情况选择合适的方法）
        from skimage.transform import resize
        true_model = resize(true_model, velocity_model.shape, anti_aliasing=True)
        print(f"警告：已调整真实模型尺寸从 {true_model.shape} 到 {velocity_model.shape}")
    
    # 7. 计算评估指标（现在尺寸已匹配）
    try:
        mse = mean_squared_error(true_model, velocity_model)
        ssim = structural_similarity(true_model, velocity_model, 
                                   data_range=2,  # 值域[-1,1]所以data_range=2
                                   win_size=11)   # 适当减小窗口大小
        snr = 10 * np.log10(np.sum(true_model**2) / np.sum((true_model - velocity_model)**2))
        
        print("\n===== 评估指标 ======")
        print(f"MSE: {mse:.6f}")
        print(f"SSIM: {ssim:.4f}")
        print(f"SNR: {snr:.2f} dB")
        
        with open("evaluation_metrics.txt", "w") as f:
            f.write(f"MSE: {mse:.6f}\nSSIM: {ssim:.4f}\nSNR: {snr:.2f} dB\n")
    except Exception as e:
        print(f"指标计算失败: {str(e)}")

    # 8. 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 动态确定显示范围
    vmin = min(true_model.min(), velocity_model.min())
    vmax = max(true_model.max(), velocity_model.max())
    
    im1 = ax1.imshow(true_model, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    ax1.set_title("True Model")
    fig.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(velocity_model, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_title(f"Predicted Model\nMSE={mse:.4f}" if 'mse' in locals() else "Predicted Model")
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('result_comparison.png', dpi=300)
    plt.show()
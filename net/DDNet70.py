# -*- coding: utf-8 -*-
"""
Build network

Created on Feb 2023

@author: Xing-Yi Zhang (zxy20004182@163.com)

"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, structural_similarity
from torchsummary import summary
from skimage.transform import resize

class SeismicRecordDownSampling(nn.Module):
    '''
    Downsampling module for seismic records
    '''
    def __init__(self, shot_num):
        super().__init__()

        self.pre_dim_reducer1 = ConvBlock(shot_num, 8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer2 = ConvBlock(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer3 = ConvBlock(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer4 = ConvBlock(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer5 = ConvBlock(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer6 = ConvBlock(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):

        width = x.shape[3]
        new_size = [width * 8, width]
        dimred0 = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)

        dimred1 = self.pre_dim_reducer1(dimred0)
        dimred2 = self.pre_dim_reducer2(dimred1)
        dimred3 = self.pre_dim_reducer3(dimred2)
        dimred4 = self.pre_dim_reducer4(dimred3)
        dimred5 = self.pre_dim_reducer5(dimred4)
        dimred6 = self.pre_dim_reducer6(dimred5)

        return dimred6

###############################################
#         Conventional Network Unit           #
# (The red arrow shown in Fig 1 of the paper) #
###############################################

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, activ_fuc = nn.ReLU(inplace=True)):
        '''
        Conventional Network Unit
        (The red arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_batchnorm: Whether to use BN
        :param activ_fuc:    Activation function
        '''
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       activ_fuc)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

##################################################
#             Downsampling Unit                  #
# (The purple arrow shown in Fig 1 of the paper) #
##################################################

class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, activ_fuc = nn.ReLU(inplace=True)):
        '''
        Downsampling Unit
        (The purple arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_batchnorm: Whether to use BN
        :param activ_fuc:    Activation function
        '''
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm, activ_fuc)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        skip_output = self.conv(inputs)
        outputs = self.down(skip_output)
        return outputs

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv, activ_fuc=nn.ReLU(inplace=True)):
        '''
        Upsampling Unit
        (The yellow arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        :param activ_fuc:    Activation function
        '''
        super(unetUp, self).__init__()
        self.output_lim = output_lim
        self.conv = unetConv2(in_size, out_size, True, activ_fuc)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input1, input2):
        input2 = self.up(input2)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)
        return self.conv(torch.cat([input1, input2], 1))

class netUp(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv):
        '''
        Upsampling Unit
        (The yellow arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        :param activ_fuc:    Activation function
        '''
        super(netUp, self).__init__()
        self.output_lim = output_lim
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input):
        input = self.up(input)
        output = F.interpolate(input, size=self.output_lim, mode='bilinear', align_corners=False)
        return output

###################################################
# Non-square convolution with flexible definition #
#            Similar to InversionNet              #
###################################################

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, activ_fuc = nn.ReLU(inplace=True)):
        '''
        Non-square convolution with flexible definition
        (Similar to InversionNet)

        :param in_fea:       Number of channels for convolution layer input
        :param out_fea:      Number of channels for convolution layer output
        :param kernel_size:  Size of the convolution kernel
        :param stride:       Convolution stride
        :param padding:      Convolution padding
        :param activ_fuc:    Activation function
        '''
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(activ_fuc)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

###################################################
#    Convolution at the end for normalization     #
#            Similar to InversionNet              #
###################################################

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        '''
        Convolution at the end for normalization
        (Similar to InversionNet)

        :param in_fea:       Number of channels for convolution layer input
        :param out_fea:      Number of channels for convolution layer output
        :param kernel_size:  Size of the convolution kernel
        :param stride:       Convolution stride
        :param padding:      Convolution padding
        '''
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LossDDNet:
    def __init__(self, weights=[1, 1], entropy_weight=[1, 1]):
        '''
        Define the loss function of DDNet

        :param weights:         The weights of the two decoders in the calculation of the loss value.
        :param entropy_weight:  The weights of the two output channels in the second decoder.
        '''

        self.criterion1 = nn.MSELoss()
        ew = torch.from_numpy(np.array(entropy_weight).astype(np.float32)).cuda()
        self.criterion2 = nn.CrossEntropyLoss(weight = ew)    # For multi-classification, the current issue is a binary problem (either black or white).
        self.weights = weights

    def __call__(self, outputs1, outputs2, targets1, targets2):
        '''

        :param outputs1: Output of the first decoder
        :param outputs2: Velocity model
        :param targets1: Output of the second decoder
        :param targets2: Profile of the speed model
        :return:
        '''
        mse = self.criterion1(outputs1, targets1)
        cross = self.criterion2(outputs2, torch.squeeze(targets2).long())

        criterion = (self.weights[0] * mse + self.weights[1] * cross)

        return criterion

############################################
#          DD-Net70 Architecture           #
############################################

class DDNet70Model(nn.Module):
    def __init__(self, n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True):
        '''
        DD-Net70 Architecture

        :param n_classes:    Number of channels of output (any single decoder)
        :param in_channels:  Number of channels of network input
        :param is_deconv:    Whether to use deconvolution
        :param is_batchnorm: Whether to use BN
        '''
        super(DDNet70Model, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        self.pre_seis_conv = SeismicRecordDownSampling(self.in_channels)

        # Intrinsic UNet section
        self.down3 = unetDown(32, 64, self.is_batchnorm)
        self.down4 = unetDown(64, 128, self.is_batchnorm)
        self.down5 = unetDown(128, 256, self.is_batchnorm)

        self.center = unetDown(256, 512, self.is_batchnorm)

        self.up5 = unetUp(512, 256, output_lim=[9, 9], is_deconv=self.is_deconv)
        self.up4 = unetUp(256, 128, output_lim=[18, 18], is_deconv=self.is_deconv)
        self.up3 = netUp(128, 64, output_lim=[35, 35], is_deconv=self.is_deconv)
        self.up2 = netUp(64, 32, output_lim=[70, 70], is_deconv=self.is_deconv)

        self.dc1_final = ConvBlock_Tanh(32, self.n_classes)
        self.dc2_final = ConvBlock_Tanh(32, 2)

    def forward(self, inputs, _=None):
        '''

        :param inputs:      Input Image
        :param _:           Variables for filling, for alignment with DD-Net
        :return:
        '''

        compress_seis = self.pre_seis_conv(inputs)

        down3 = self.down3(compress_seis)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        center = self.center(down5)

        # 16*18*512
        decoder1_image = center
        decoder2_image = center

        #################
        ###  Decoder1 ###
        #################
        dc1_up5 = self.up5(down5, decoder1_image)
        dc1_up4 = self.up4(down4, dc1_up5)
        dc1_up3 = self.up3(dc1_up4)
        dc1_up2 = self.up2(dc1_up3)


        #################
        ###  Decoder2 ###
        #################
        dc2_up5 = self.up5(down5, decoder2_image)
        dc2_up4 = self.up4(down4, dc2_up5)
        dc2_up3 = self.up3(dc2_up4)
        dc2_up2 = self.up2(dc2_up3)


        return [self.dc1_final(dc1_up2), self.dc2_final(dc2_up2)]

############################################
#          SD-Net70 Architecture           #
############################################

class SDNet70Model(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        '''
        DD-Net70 Architecture

        :param n_classes:    Number of channels of output (any single decoder)
        :param in_channels:  Number of channels of network input
        :param is_deconv:    Whether to use deconvolution
        :param is_batchnorm: Whether to use BN
        '''
        super(SDNet70Model, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        self.pre_seis_conv = SeismicRecordDownSampling(self.in_channels)

        # Intrinsic UNet section
        self.down3 = unetDown(32, 64, self.is_batchnorm)
        self.down4 = unetDown(64, 128, self.is_batchnorm)
        self.down5 = unetDown(128, 256, self.is_batchnorm)

        self.center = unetDown(256, 512, self.is_batchnorm)

        self.up5 = unetUp(512, 256, output_lim=[9, 9], is_deconv=self.is_deconv)
        self.up4 = unetUp(256, 128, output_lim=[18, 18], is_deconv=self.is_deconv)
        self.up3 = netUp(128, 64, output_lim=[35, 35], is_deconv=self.is_deconv)
        self.up2 = netUp(64, 32, output_lim=[70, 70], is_deconv=self.is_deconv)

        self.dc1_final = ConvBlock_Tanh(32, self.n_classes)
        self.dc2_final = ConvBlock_Tanh(32, 2)

    def forward(self, inputs, _=None):
        '''

        :param inputs:      Input Image
        :param _:           Variables for filling, for alignment with DD-Net
        :return:
        '''

        compress_seis = self.pre_seis_conv(inputs)

        down3 = self.down3(compress_seis)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        center = self.center(down5)

        # 16*18*512
        decoder1_image = center
        decoder2_image = center

        #################
        ###  Decoder1 ###
        #################
        dc1_up5 = self.up5(down5, decoder1_image)
        dc1_up4 = self.up4(down4, dc1_up5)
        dc1_up3 = self.up3(dc1_up4)
        dc1_up2 = self.up2(dc1_up3)

        return self.dc1_final(dc1_up2)

if __name__ == '__main__':
    # 1. 初始化模型
    model = DDNet70Model(n_classes=1, in_channels=7, is_deconv=True, is_batchnorm=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 2. 打印模型结构
    print("\n" + "="*50 + " Model Architecture " + "="*50)
    summary(model, input_size=[(7, 1000, 70)], device=device.type)
    
    # 3. 生成模拟数据
    true_model = np.zeros((1000, 70))
    true_model[300:700, 20:50] = np.linspace(0.1, 0.8, 400)[:, np.newaxis]
    true_model += np.random.normal(0, 0.02, size=true_model.shape)
    
    # 4. 创建输入数据
    input_data = torch.randn(1, 7, 1000, 70).to(device)
    
    # 5. 模型推理（关键修改：处理列表输出）
    with torch.no_grad():
        output = model(input_data)
        
        # 处理多输出情况
        if isinstance(output, list) or isinstance(output, tuple):
            print(f"模型返回了{len(output)}个输出，使用第一个输出作为速度模型")
            velocity_tensor = output[0]  # 取第一个输出
        else:
            velocity_tensor = output
            
        # 确保获取的是张量
        velocity_model = velocity_tensor[0, 0].cpu().numpy()  # 现在可以安全使用[0,0]
    
    # 6. 尺寸匹配
    if true_model.shape != velocity_model.shape:
        velocity_model = resize(velocity_model, true_model.shape, anti_aliasing=True)
        print(f"调整尺寸：{velocity_model.shape} → {true_model.shape}")
    
    # 7. 计算评估指标
    mse = mean_squared_error(true_model, velocity_model)
    ssim = structural_similarity(true_model, velocity_model, 
                               data_range=true_model.max()-true_model.min())
    snr = 10 * np.log10(np.sum(true_model**2) / np.sum((true_model - velocity_model)**2))
    
    # 8. 可视化
    plt.figure(figsize=(18, 6))
    
    # 真实模型
    plt.subplot(1, 3, 1)
    plt.imshow(true_model, cmap='seismic', aspect='auto', 
              vmin=0, vmax=0.8, extent=[0, 70, 1000, 0])
    plt.colorbar(label='Velocity (km/s)')
    plt.title("True Model")
    
    # 预测模型
    plt.subplot(1, 3, 2)
    plt.imshow(velocity_model, cmap='seismic', aspect='auto',
              vmin=0, vmax=0.8, extent=[0, 70, 1000, 0])
    plt.colorbar(label='Velocity (km/s)')
    plt.title(f"Predicted\nMSE={mse:.4f}, SSIM={ssim:.4f}")
    
    # 残差图
    plt.subplot(1, 3, 3)
    residual = true_model - velocity_model
    plt.imshow(residual, cmap='coolwarm', aspect='auto',
              vmin=-0.3, vmax=0.3, extent=[0, 70, 1000, 0])
    plt.colorbar(label='Residual (km/s)')
    plt.title(f"Residual\nSNR={snr:.2f} dB")
    
    plt.tight_layout()
    plt.savefig('result.png', dpi=300)
    plt.show()

    # 9. 保存评估结果
    report = f"""
=== 评估结果 ===
输入尺寸: 7×1000×70
输出尺寸: {velocity_model.shape}
MSE (越小越好): {mse:.5f}
SSIM (1.0最佳): {ssim:.4f}
SNR (越大越好): {snr:.2f} dB
真实速度范围: {true_model.min():.2f}~{true_model.max():.2f} km/s
预测速度范围: {velocity_model.min():.2f}~{velocity_model.max():.2f} km/s
"""
    print(report)
    with open("evaluation.txt", "w") as f:
        f.write(report)

#运行结果
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 8, 280, 70]             176
#        BatchNorm2d-2           [-1, 8, 280, 70]              16
#               ReLU-3           [-1, 8, 280, 70]               0
#               ReLU-4           [-1, 8, 280, 70]               0
#               ReLU-5           [-1, 8, 280, 70]               0
#               ReLU-6           [-1, 8, 280, 70]               0
#               ReLU-7           [-1, 8, 280, 70]               0
#               ReLU-8           [-1, 8, 280, 70]               0
#          ConvBlock-9           [-1, 8, 280, 70]               0
#            Conv2d-10           [-1, 8, 280, 70]             584
#       BatchNorm2d-11           [-1, 8, 280, 70]              16
#              ReLU-12           [-1, 8, 280, 70]               0
#              ReLU-13           [-1, 8, 280, 70]               0
#              ReLU-14           [-1, 8, 280, 70]               0
#              ReLU-15           [-1, 8, 280, 70]               0
#              ReLU-16           [-1, 8, 280, 70]               0
#              ReLU-17           [-1, 8, 280, 70]               0
#         ConvBlock-18           [-1, 8, 280, 70]               0
#            Conv2d-19          [-1, 16, 140, 70]             400
#       BatchNorm2d-20          [-1, 16, 140, 70]              32
#              ReLU-21          [-1, 16, 140, 70]               0
#              ReLU-22          [-1, 16, 140, 70]               0
#              ReLU-23          [-1, 16, 140, 70]               0
#              ReLU-24          [-1, 16, 140, 70]               0
#              ReLU-25          [-1, 16, 140, 70]               0
#              ReLU-26          [-1, 16, 140, 70]               0
#         ConvBlock-27          [-1, 16, 140, 70]               0
#            Conv2d-28          [-1, 16, 140, 70]           2,320
#       BatchNorm2d-29          [-1, 16, 140, 70]              32
#              ReLU-30          [-1, 16, 140, 70]               0
#              ReLU-31          [-1, 16, 140, 70]               0
#              ReLU-32          [-1, 16, 140, 70]               0
#              ReLU-33          [-1, 16, 140, 70]               0
#              ReLU-34          [-1, 16, 140, 70]               0
#              ReLU-35          [-1, 16, 140, 70]               0
#         ConvBlock-36          [-1, 16, 140, 70]               0
#            Conv2d-37           [-1, 32, 70, 70]           1,568
#       BatchNorm2d-38           [-1, 32, 70, 70]              64
#              ReLU-39           [-1, 32, 70, 70]               0
#              ReLU-40           [-1, 32, 70, 70]               0
#              ReLU-41           [-1, 32, 70, 70]               0
#              ReLU-42           [-1, 32, 70, 70]               0
#              ReLU-43           [-1, 32, 70, 70]               0
#              ReLU-44           [-1, 32, 70, 70]               0
#         ConvBlock-45           [-1, 32, 70, 70]               0
#            Conv2d-46           [-1, 32, 70, 70]           9,248
#       BatchNorm2d-47           [-1, 32, 70, 70]              64
#              ReLU-48           [-1, 32, 70, 70]               0
#              ReLU-49           [-1, 32, 70, 70]               0
#              ReLU-50           [-1, 32, 70, 70]               0
#              ReLU-51           [-1, 32, 70, 70]               0
#              ReLU-52           [-1, 32, 70, 70]               0
#              ReLU-53           [-1, 32, 70, 70]               0
#         ConvBlock-54           [-1, 32, 70, 70]               0
# SeismicRecordDownSampling-55           [-1, 32, 70, 70]               0
#            Conv2d-56           [-1, 64, 70, 70]          18,496
#       BatchNorm2d-57           [-1, 64, 70, 70]             128
#              ReLU-58           [-1, 64, 70, 70]               0
#              ReLU-59           [-1, 64, 70, 70]               0
#              ReLU-60           [-1, 64, 70, 70]               0
#              ReLU-61           [-1, 64, 70, 70]               0
#              ReLU-62           [-1, 64, 70, 70]               0
#              ReLU-63           [-1, 64, 70, 70]               0
#              ReLU-64           [-1, 64, 70, 70]               0
#              ReLU-65           [-1, 64, 70, 70]               0
#            Conv2d-66           [-1, 64, 70, 70]          36,928
#       BatchNorm2d-67           [-1, 64, 70, 70]             128
#              ReLU-68           [-1, 64, 70, 70]               0
#              ReLU-69           [-1, 64, 70, 70]               0
#              ReLU-70           [-1, 64, 70, 70]               0
#              ReLU-71           [-1, 64, 70, 70]               0
#              ReLU-72           [-1, 64, 70, 70]               0
#              ReLU-73           [-1, 64, 70, 70]               0
#              ReLU-74           [-1, 64, 70, 70]               0
#              ReLU-75           [-1, 64, 70, 70]               0
#         unetConv2-76           [-1, 64, 70, 70]               0
#         MaxPool2d-77           [-1, 64, 35, 35]               0
#          unetDown-78           [-1, 64, 35, 35]               0
#            Conv2d-79          [-1, 128, 35, 35]          73,856
#       BatchNorm2d-80          [-1, 128, 35, 35]             256
#              ReLU-81          [-1, 128, 35, 35]               0
#              ReLU-82          [-1, 128, 35, 35]               0
#              ReLU-83          [-1, 128, 35, 35]               0
#              ReLU-84          [-1, 128, 35, 35]               0
#              ReLU-85          [-1, 128, 35, 35]               0
#              ReLU-86          [-1, 128, 35, 35]               0
#              ReLU-87          [-1, 128, 35, 35]               0
#              ReLU-88          [-1, 128, 35, 35]               0
#            Conv2d-89          [-1, 128, 35, 35]         147,584
#       BatchNorm2d-90          [-1, 128, 35, 35]             256
#              ReLU-91          [-1, 128, 35, 35]               0
#              ReLU-92          [-1, 128, 35, 35]               0
#              ReLU-93          [-1, 128, 35, 35]               0
#              ReLU-94          [-1, 128, 35, 35]               0
#              ReLU-95          [-1, 128, 35, 35]               0
#              ReLU-96          [-1, 128, 35, 35]               0
#              ReLU-97          [-1, 128, 35, 35]               0
#              ReLU-98          [-1, 128, 35, 35]               0
#         unetConv2-99          [-1, 128, 35, 35]               0
#        MaxPool2d-100          [-1, 128, 18, 18]               0
#         unetDown-101          [-1, 128, 18, 18]               0
#           Conv2d-102          [-1, 256, 18, 18]         295,168
#      BatchNorm2d-103          [-1, 256, 18, 18]             512
#             ReLU-104          [-1, 256, 18, 18]               0
#             ReLU-105          [-1, 256, 18, 18]               0
#             ReLU-106          [-1, 256, 18, 18]               0
#             ReLU-107          [-1, 256, 18, 18]               0
#             ReLU-108          [-1, 256, 18, 18]               0
#             ReLU-109          [-1, 256, 18, 18]               0
#             ReLU-110          [-1, 256, 18, 18]               0
#             ReLU-111          [-1, 256, 18, 18]               0
#           Conv2d-112          [-1, 256, 18, 18]         590,080
#      BatchNorm2d-113          [-1, 256, 18, 18]             512
#             ReLU-114          [-1, 256, 18, 18]               0
#             ReLU-115          [-1, 256, 18, 18]               0
#             ReLU-116          [-1, 256, 18, 18]               0
#             ReLU-117          [-1, 256, 18, 18]               0
#             ReLU-118          [-1, 256, 18, 18]               0
#             ReLU-119          [-1, 256, 18, 18]               0
#             ReLU-120          [-1, 256, 18, 18]               0
#             ReLU-121          [-1, 256, 18, 18]               0
#        unetConv2-122          [-1, 256, 18, 18]               0
#        MaxPool2d-123            [-1, 256, 9, 9]               0
#         unetDown-124            [-1, 256, 9, 9]               0
#           Conv2d-125            [-1, 512, 9, 9]       1,180,160
#      BatchNorm2d-126            [-1, 512, 9, 9]           1,024
#             ReLU-127            [-1, 512, 9, 9]               0
#             ReLU-128            [-1, 512, 9, 9]               0
#             ReLU-129            [-1, 512, 9, 9]               0
#             ReLU-130            [-1, 512, 9, 9]               0
#             ReLU-131            [-1, 512, 9, 9]               0
#             ReLU-132            [-1, 512, 9, 9]               0
#             ReLU-133            [-1, 512, 9, 9]               0
#             ReLU-134            [-1, 512, 9, 9]               0
#           Conv2d-135            [-1, 512, 9, 9]       2,359,808
#      BatchNorm2d-136            [-1, 512, 9, 9]           1,024
#             ReLU-137            [-1, 512, 9, 9]               0
#             ReLU-138            [-1, 512, 9, 9]               0
#             ReLU-139            [-1, 512, 9, 9]               0
#             ReLU-140            [-1, 512, 9, 9]               0
#             ReLU-141            [-1, 512, 9, 9]               0
#             ReLU-142            [-1, 512, 9, 9]               0
#             ReLU-143            [-1, 512, 9, 9]               0
#             ReLU-144            [-1, 512, 9, 9]               0
#        unetConv2-145            [-1, 512, 9, 9]               0
#        MaxPool2d-146            [-1, 512, 5, 5]               0
#         unetDown-147            [-1, 512, 5, 5]               0
#  ConvTranspose2d-148          [-1, 256, 10, 10]         524,544
#           Conv2d-149            [-1, 256, 9, 9]       1,179,904
#      BatchNorm2d-150            [-1, 256, 9, 9]             512
#             ReLU-151            [-1, 256, 9, 9]               0
#             ReLU-152            [-1, 256, 9, 9]               0
#             ReLU-153            [-1, 256, 9, 9]               0
#             ReLU-154            [-1, 256, 9, 9]               0
#           Conv2d-155            [-1, 256, 9, 9]         590,080
#      BatchNorm2d-156            [-1, 256, 9, 9]             512
#             ReLU-157            [-1, 256, 9, 9]               0
#             ReLU-158            [-1, 256, 9, 9]               0
#             ReLU-159            [-1, 256, 9, 9]               0
#             ReLU-160            [-1, 256, 9, 9]               0
#        unetConv2-161            [-1, 256, 9, 9]               0
#           unetUp-162            [-1, 256, 9, 9]               0
#  ConvTranspose2d-163          [-1, 128, 18, 18]         131,200
#           Conv2d-164          [-1, 128, 18, 18]         295,040
#      BatchNorm2d-165          [-1, 128, 18, 18]             256
#             ReLU-166          [-1, 128, 18, 18]               0
#             ReLU-167          [-1, 128, 18, 18]               0
#             ReLU-168          [-1, 128, 18, 18]               0
#             ReLU-169          [-1, 128, 18, 18]               0
#           Conv2d-170          [-1, 128, 18, 18]         147,584
#      BatchNorm2d-171          [-1, 128, 18, 18]             256
#             ReLU-172          [-1, 128, 18, 18]               0
#             ReLU-173          [-1, 128, 18, 18]               0
#             ReLU-174          [-1, 128, 18, 18]               0
#             ReLU-175          [-1, 128, 18, 18]               0
#        unetConv2-176          [-1, 128, 18, 18]               0
#           unetUp-177          [-1, 128, 18, 18]               0
#  ConvTranspose2d-178           [-1, 64, 36, 36]          32,832
#            netUp-179           [-1, 64, 35, 35]               0
#  ConvTranspose2d-180           [-1, 32, 70, 70]           8,224
#            netUp-181           [-1, 32, 70, 70]               0
#  ConvTranspose2d-182          [-1, 256, 10, 10]         524,544
#           Conv2d-183            [-1, 256, 9, 9]       1,179,904
#      BatchNorm2d-184            [-1, 256, 9, 9]             512
#             ReLU-185            [-1, 256, 9, 9]               0
#             ReLU-186            [-1, 256, 9, 9]               0
#             ReLU-187            [-1, 256, 9, 9]               0
#             ReLU-188            [-1, 256, 9, 9]               0
#           Conv2d-189            [-1, 256, 9, 9]         590,080
#      BatchNorm2d-190            [-1, 256, 9, 9]             512
#             ReLU-191            [-1, 256, 9, 9]               0
#             ReLU-192            [-1, 256, 9, 9]               0
#             ReLU-193            [-1, 256, 9, 9]               0
#             ReLU-194            [-1, 256, 9, 9]               0
#        unetConv2-195            [-1, 256, 9, 9]               0
#           unetUp-196            [-1, 256, 9, 9]               0
#  ConvTranspose2d-197          [-1, 128, 18, 18]         131,200
#           Conv2d-198          [-1, 128, 18, 18]         295,040
#      BatchNorm2d-199          [-1, 128, 18, 18]             256
#             ReLU-200          [-1, 128, 18, 18]               0
#             ReLU-201          [-1, 128, 18, 18]               0
#             ReLU-202          [-1, 128, 18, 18]               0
#             ReLU-203          [-1, 128, 18, 18]               0
#           Conv2d-204          [-1, 128, 18, 18]         147,584
#      BatchNorm2d-205          [-1, 128, 18, 18]             256
#             ReLU-206          [-1, 128, 18, 18]               0
#             ReLU-207          [-1, 128, 18, 18]               0
#             ReLU-208          [-1, 128, 18, 18]               0
#             ReLU-209          [-1, 128, 18, 18]               0
#        unetConv2-210          [-1, 128, 18, 18]               0
#           unetUp-211          [-1, 128, 18, 18]               0
#  ConvTranspose2d-212           [-1, 64, 36, 36]          32,832
#            netUp-213           [-1, 64, 35, 35]               0
#  ConvTranspose2d-214           [-1, 32, 70, 70]           8,224
#            netUp-215           [-1, 32, 70, 70]               0
#           Conv2d-216            [-1, 1, 70, 70]             289
#      BatchNorm2d-217            [-1, 1, 70, 70]               2
#             Tanh-218            [-1, 1, 70, 70]               0
#   ConvBlock_Tanh-219            [-1, 1, 70, 70]               0
#           Conv2d-220            [-1, 2, 70, 70]             578
#      BatchNorm2d-221            [-1, 2, 70, 70]               4
#             Tanh-222            [-1, 2, 70, 70]               0
#   ConvBlock_Tanh-223            [-1, 2, 70, 70]               0
# ================================================================
# Total params: 10,543,201
# Trainable params: 10,543,201
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 1.87
# Forward/backward pass size (MB): 185.44
# Params size (MB): 40.22
# Estimated Total Size (MB): 227.53
# ----------------------------------------------------------------
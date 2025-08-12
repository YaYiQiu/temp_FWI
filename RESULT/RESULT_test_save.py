# -*- coding: utf-8 -*-
"""
Created on 2023/10/25 9:24

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
################################################
########        IMPORT LIBARIES         ########
################################################
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
from PathConfig import *
from model.InversionNet import *
from model.FCNVMB import *
from model.IAEDN import *
from data.data import *
from data.show import *
################################################
########         LOAD    NETWORK        ########
################################################

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
NoiseFlag = True  #True False
DataSet = 'FlatFaultA/'  # CurveVelA  FlatFaultA  FlatVelA   CurveFaultA   marmousi_70_70
# model = 'Q' # InversionNet  VelocityGAN  IAEDN_WTUU   Noise_IAEDN_WTUU
PreModelname = 'IADNE_WTUU_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch160.pkl'
model_file = 'D:/Xu-Qiong/ABA-FWI_v1/train_result/'+ DataSet + PreModelname
net = IAEDN_WTUU()   #InversionNet  IAEDN_WTUU Inversion_U
net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))


# FlatVelA
# InversionNet_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch120.pkl
# VelocityGAN_TrainSize24000_Epoch500_BatchSize20_LR0.0001_epoch445.pkl
# IAEDN_WTUU_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch120.pkl
# Noise_IAEDN_WTUU_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch120.pkl

# FlatFaultA
# InversionNet_TrainSize48000_Epoch200_BatchSize20_LR0.0001_epoch120.pkl
# VelocityGAN_TrainSize48000_Epoch500_BatchSize20_LR0.0001_epoch420.pkl
# IADNE_WTUU_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch160.pkl
# Noise_IAEDN_WTUU_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch140.pkl

# NoNet_IADNE_WTUU_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch90.pkl
# NoLoss_IAEDN_WTUU_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch120.pkl

# CurveVelA
# InversionNet_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch90.pkl
# VelocityGAN_TrainSize24000_Epoch180_BatchSize20_LR0.0001_epoch80.pkl
# IAEDN_WTUU_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch120.pkl
# Noise_IAEDN_WTUU_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch140.pkl

#  CurveFaultA
# InversionNet_TrainSize48000_Epoch200_BatchSize20_LR0.0001_epoch120.pkl
# VelocityGAN_TrainSize48000_Epoch500_BatchSize20_LR0.0001_epoch400.pkl
# IAEDN_WTUU_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch130.pkl
# Noise_IAEDN_WTUU_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch100.pkl

# NoNet_IADNE_WTUU_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch100.pkl
# NoLoss_IAEDN_WTUU_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch120.pkl
path =  'C:/Users/DELL/Desktop/OpenFWI_RESULTS/' + DataSet
Data_path = 'D:/Data/OpenFWI/' + DataSet    #  D:/Data/OpenFWI/  D:/Data/Marmousi/
################################################
########    LOADING TESTING DATA       ########
################################################

print('***************** 正在加载测试数据集 *****************')
TestSize = 500
TestBatchSize = 500
# 输出当前工作目录
dataset_dir = Data_path

testSet = Dataset_openfwi_test(dataset_dir, TestSize, 1, "seismic", "test")   # 11 for test  # dataset_dir, TestSize, 1, "seismic", "test"

test_loader = DataLoader(testSet, batch_size=TestBatchSize, shuffle=False)

################################################
########            TESTING             ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('                  开始测试                  ')
print('*******************************************')
print('*******************************************')
print()

# Initialization
since = time.time()

Total_PSNR = np.zeros((1, TestSize), dtype=float)
Total_SSIM = np.zeros((1, TestSize), dtype=float)
Total_MSE = np.zeros((1, TestSize), dtype=float)
Total_MAE = np.zeros((1, TestSize), dtype=float)
Total_UQI = np.zeros((1, TestSize), dtype=float)
Total_LPIPS = np.zeros((1, TestSize), dtype=float)


Prediction = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
GT = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
Prediction_N = np.zeros((6, ModelDim[0], ModelDim[1]), dtype=float)
GT_N = np.zeros((6, ModelDim[0], ModelDim[1]), dtype=float)

total = 0

for i, (seismic_datas, vmodels, vmodel_max_min) in enumerate(test_loader):
    # Predictions
    net.eval()
    net.to(device)
    vmodels = vmodels[0].to(device)
    seismic_datas = seismic_datas[0].to(device)
    vmodel_max_min = vmodel_max_min[0].to(device)

    if NoiseFlag:
        # 添加高斯噪声
        seed = 42
        torch.manual_seed(seed)

        noise_mean = 0
        noise_std = 0.1
        noise = torch.normal(mean=noise_mean, std=noise_std, size=seismic_datas.shape).to(device)
        seismic_datas = seismic_datas + noise

    # Forward prediction
    outputs = net(seismic_datas)

    # 转换参数的输入类型为 float
    # outputs = outputs.to(torch.float32)
    # vmodels = vmodels.to(torch.float32)

    outputs = outputs.data.cpu().numpy()
    outputs = np.where(outputs > 0.0, outputs, 0.0)

    gts = vmodels.data.cpu().numpy()
    vmodel_max_min = vmodel_max_min.data.cpu().numpy()

    m = 0
    # Calculate the PSNR, SSIM
    for k in range(TestBatchSize):
        pd = outputs[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        gt = gts[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        vmax = vmodel_max_min[k, 0]
        vmin = vmodel_max_min[k, 1]
        if total in [46, 60, 2]  :    #FlatFaultA[46, 60, 2][3,8,15,31] CurveFaultA[0, 5466, 5868][2, 8, 10, 14]  FlatVelA[0, 2, 5][22, 44] CurveVelA[0,2,3][22,26,48,49]
            # 消融FlatFaultA[46, 60, 2] CurveFaultA[26, 27, 34, 43, 46, 59]
            # 消融实验 大修  FlatFaultA[46, 60, 2]46    CurveFaultA[3,5,11]   11
            pd_N = pd * (vmax - vmin) + vmin
            gt_N = gt * (vmax - vmin) + vmin
            # pain_openfwi_velocity_model1(gt_N, vmin, vmax)
            pain_openfwi_velocity_model1(pd_N,vmin,vmax)
            # pain_openfwi_velocity_model1(pd_N - gt_N,-100,100)
            Prediction_N[m, :, :] = pd_N
            GT_N[m, :, :] = gt_N
            m = m + 1

            psnr = PSNR(gt, pd)
            ssim = SSIM(gt, pd)
            mse = MSE(pd, gt)
            mae = MAE(pd, gt)
            uqi = UQI(pd, gt)
            lpips = LPIPS(pd, gt)
            print('The %d testing psnr: %.2f, SSIM: %.4f, MSE:  %.4f, MAE:  %.4f, UQI:  %.4f, LPIPS: %.4f' % (m, psnr, ssim, mse, mae, uqi, lpips))
        total = total + 1


    #SaveSelectedTestResults(Prediction_N, GT_N, model, path)  #InversionNet


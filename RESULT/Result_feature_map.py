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
from matplotlib import gridspec
################################################
########         LOAD    NETWORK        ########
################################################

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')


model_file = train_result_dir + PreModelname
net = Inversion_U()   #InversionNet  IAEDN  FCNVMB  IAEDN_WTUU  Inversion_U
net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

################################################
########    LOADING TESTING DATA       ########
################################################

print('***************** 正在加载测试数据集 *****************')
TestSize = 500
TestBatchSize = 200
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
Prediction_N = np.zeros((4, ModelDim[0], ModelDim[1]), dtype=float)
GT_N = np.zeros((4, ModelDim[0], ModelDim[1]), dtype=float)

total = 0

for i, (seismic_datas, vmodels, vmodel_max_min) in enumerate(test_loader):
    # Predictions
    net.eval()
    net.to(device)
    vmodels = vmodels[0].to(device)
    seismic_datas = seismic_datas[0].to(device)
    vmodel_max_min = vmodel_max_min[0].to(device)

    # Forward prediction
    outputs = net(seismic_datas)
    outputs_np = [output.data.cpu().numpy() for output in outputs]


    for m in range(2):
        for j in range(TestBatchSize):
            if j in [43]: # 消融FlatFaultA[46, 60, 2]60 CurveFaultA[26, 27, 34, 43, 46, 59]43
                for k in range(12):
                    fig = plt.figure(figsize=(6, 6), dpi=50)
                    plt.imshow(outputs_np[m][j][k], interpolation='nearest')
                    plt.axis('off')  # 可以选择关闭坐标轴
                    plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
                    # plt.show()
                    fig.savefig("./feature_map/FlatFaultA(ABA-FWI)/" + str(j) + '_' +str(m) + '_' +str(k) + '.png')
                    plt.close()
                    # print("ok")
    break
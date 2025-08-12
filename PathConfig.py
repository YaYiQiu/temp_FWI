# -*- coding: utf-8 -*-
"""
Created on 2023/10/17 11:01

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
import os
from ParamConfig import *

###################################################
####                DATA   PATHS              #####
###################################################
Data_dir = 'D:/Data/OpenFWI/'  # 数据所在目录 D:/Data/OpenFWI/  D:/Data/Marmousi/
DataSet = Dataset_name + '/'
Data_path = Data_dir + DataSet

###################################################
####            RESULT   PATHS                #####
###################################################
main_dir = 'D:/Xu-Qiong/ABA-FWI_v1/'  # 当前工作目录

# Check the main directory
if len(main_dir) == 0:
    raise Exception('Please specify path to correct directory!')

# Save training result
if os.path.exists('train_result/' + DataSet):
    train_result_dir = main_dir + 'train_result/' + DataSet  # Replace your data path here
else:
    os.makedirs('train_result/' + DataSet)
    train_result_dir = main_dir + 'train_result/' + DataSet

# Save testing result
if os.path.exists('test_result/' + DataSet):
    test_result_dir = main_dir + 'test_result/' + DataSet  # Replace your data path here
else:
    os.makedirs('test_result/' + DataSet)
    test_result_dir = main_dir + 'test_result/' + DataSet

####################################################
####                   FileName                #####
####################################################
# You can set model name for training.
# For comparative experiment. InversionNet  VelocityGAN
# For ablation experiment.  ABA-Net ABA-Loss ABA-FWI
NoiseFlag = False  # If True add noise.
modelName = 'ABA-FWI'
# VelocityGAN | InversionNet | DD-Net70
# ABA-Net | ABA-Loss | ABA-FWI

tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch' + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
tagM4 = '_LR' + str(LearnRate)

ModelName = modelName + tagM1 + tagM2 + tagM3 + tagM4

# Load pre-trained model
PreModelname = ''

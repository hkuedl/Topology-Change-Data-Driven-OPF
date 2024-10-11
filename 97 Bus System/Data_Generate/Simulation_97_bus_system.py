import matplotlib.pyplot as plt
import numpy as np
from pyomo.core import *
import pyomo.kernel as pyo
from pypower.api import case30,ppoption, runopf, runpf, case118, case14, case9, case4gs
import copy
# from case33bw import *
import time
import random
import os
import torch

def seed_torch(seed=114514): # 114544
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

data_in = case118()
data_in2 = case14()

import scipy.io as io
mat_path = 'mpc_ATS_97.mat'
data = io.loadmat(mat_path)

data_in['version'] = data['version']
data_in['baseMVA'] = data['baseMVA']
data_in['bus'] = data['bus']
data_in['gen'] = data['gen']
data_in['branch'] = data['branch']
data_in['gencost'] = data['gencost']

data_in['version'] = '2'
data_in['baseMVA'] = 100.0
data_in['bus'] = data_in['bus'][:,0:13]
data_in['gen'] = data_in['gen'][:,0:21]
data_in['branch'] = data_in['branch'][:,0:13]

Add_Line = copy.deepcopy(data_in['branch'][125:126,0:21])
Add_Line[0,0] = 24.0
Add_Line[0,1] = 623.0
Add_Line[0,2] = 0.03446
Add_Line[0,3] = 0.08070
Add_Line[0,4] = 0.01110

# Add_Line2 = copy.deepcopy(data_in['branch'][125:126,0:21])
# Add_Line2[0,0] = 451.0
# Add_Line2[0,1] = 709.0
# Add_Line2[0,2] = 0.12218
# Add_Line2[0,3] = 0.24870
# Add_Line2[0,4] = 0.03419
#
# Add_Line3 = copy.deepcopy(data_in['branch'][125:126,0:21])
# Add_Line3[0,0] = 377.0
# Add_Line3[0,1] = 832.0
# Add_Line3[0,2] = 0.05025
# Add_Line3[0,3] = 0.10229
# Add_Line3[0,4] = 0.01406
#
Add_Line4 = copy.deepcopy(data_in['branch'][125:126,0:21])
Add_Line4[0,0] = 202.0
Add_Line4[0,1] = 705.0
Add_Line4[0,2] = 0.08869
Add_Line4[0,3] = 0.15054
Add_Line4[0,4] = 0.02482

data_in['branch'] = np.vstack((data_in['branch'], Add_Line, Add_Line4))

Bran_Mat = copy.deepcopy(data_in['branch'][:,2:3])
[Mat_x, Mat_y] = np.where(Bran_Mat <= 0.01)
Bran_Mat_1 = copy.deepcopy(data_in['branch'][:,2:5])
Bran_Mat_1[Mat_x,0] = 0.01000
Bran_Mat_1[Mat_x,1] = 0.03400
Bran_Mat_1[Mat_x,2] = 0.00840

data_in['branch'][:,2:5] = copy.deepcopy(Bran_Mat_1)
data_in['gencost'][:,4] = 0.01 + np.random.rand(19,)*0.05
data_in['branch'][:,5:8] = 9900.0
data_in['branch'][:,11] = -360.0
data_in['branch'][:,12] = 360.0
# data_in['bus'][:,9] = 0.0
data_in['gen'][4:9,8] = 150
ResRes, Ybus = runopf(data_in)
# %%
Load_Data_Path = 'TAS_case97_samples.mat'
Load_Data = io.loadmat(Load_Data_Path)

PD = Load_Data['Pd']
QD = Load_Data['Qd']

opt_val = Load_Data['f']

NUM = np.size(PD, 0)

Obj = np.zeros((1,1))

PD_Save = np.zeros((np.size(data_in['bus'][:,2],0),))
QD_Save = np.zeros((np.size(data_in['bus'][:,3],0),))

PGEN_Save = np.zeros((np.size(data_in['gen'][:,1],0)))
QGEN_Save = np.zeros((np.size(data_in['gen'][:,2],0)))

VM_Save = np.zeros((np.size(data_in['bus'][:,2],0)))
VA_Save = np.zeros((np.size(data_in['bus'][:,2],0)))

LMP_Save = np.zeros((np.size(data_in['bus'][:,2],0)))

[m, n] = np.where(PD < 0)
PD[m,n] = -PD[m,n]

[m1, n1] = np.where(QD < 0)
QD[m1,n1] = -QD[m1,n1]

# t1 = time.time()
# for i in range(10):
#
#     data_in['bus'][0:,2] = copy.deepcopy(np.transpose(PD[i,0:]))
#     data_in['bus'][0:,3] = copy.deepcopy(np.transpose(QD[i,0:]))
#
#     ResRes, Ybus = runopf(data_in)
#
#     if ResRes['success'] == True:
#         Obj = np.vstack((Obj, ResRes['f']))
#
#         PD_Save = np.vstack((PD_Save, data_in['bus'][:,2]))
#         QD_Save = np.vstack((QD_Save, data_in['bus'][:,3]))
#         PGEN_Save = np.vstack((PGEN_Save, ResRes['gen'][:,1]))
#         QGEN_Save = np.vstack((QGEN_Save, ResRes['gen'][:,2]))
#         VM_Save = np.vstack((VM_Save, ResRes['bus'][:,7]))
#         VA_Save = np.vstack((VA_Save, ResRes['bus'][:,8]))
#         LMP_Save = np.vstack((LMP_Save, ResRes['bus'][:,13]))
#     else:
#         exit(0)
# t2 = time.time()
# print(t2 - t1)
# %%
# Topo = 5
# Path1 = "E:\\GAN_Project\\TAS_97_bus_system\\Data_97bus_Response\\"
#
# data1 = Path1 + "PD_save_Topo_" + str(int(Topo)) + ".npy"
# data2 = Path1 + "QD_save_Topo_" + str(int(Topo)) + ".npy"
# data3 = Path1 + "P_Gen_save_Topo_" + str(int(Topo)) + ".npy"
# data4 = Path1 + "Q_Gen_save_Topo_" + str(int(Topo)) + ".npy"
# data5 = Path1 + "Vol_mag_save_Topo_" + str(int(Topo)) + ".npy"
# data6 = Path1 + "Vol_ang_save_Topo_" + str(int(Topo)) + ".npy"
# data9 = Path1 + "LMP_save_Topo_" + str(int(Topo)) + ".npy"
#
# np.save(data1, PD_Save)
# np.save(data2, QD_Save)
# np.save(data3, PGEN_Save)
# np.save(data4, QGEN_Save)
# np.save(data5, VM_Save)
# np.save(data6, VA_Save)
# np.save(data9, LMP_Save)

# %% Topo1: 451-709, 377-832
# %% Topo2: 377-832
# %% Topo3: 451-709, 377-832, 24-623
# %% Topo4: 451-709, 24-623, 202-705
# %% Topo5: 24-623, 202-705
# %% Topo6: 24-623, 377-832
# %% Topo7: 451-709, 377-832, 24-623, 202-705
# %% Topo8: 202-705

import random

import matplotlib.pyplot as plt
import numpy as np
from pyomo.core import *
import pyomo.kernel as pyo
from pypower.api import case30,ppoption, runopf, runpf, case118, case14, case9
# from makeYbus_my import *
import copy
import time

Wd1_K = 1.637
Wd2_K = 2.106
Wd3_K = 1.895

Wd1_D = 5.218
Wd2_D = 5.089
Wd3_D = 5.236

Rated_Capacity = 25

V_in = 3
V_r = 13
V_out = 25

data_in = case14()
data_in['bus'][9,1] = 2
data_in['bus'][10,1] = 2
data_in['bus'][13,1] = 2

data_in['bus'][:,2] = np.abs(data_in['bus'][:,2])
data_in['bus'][:,3] = np.abs(data_in['bus'][:,3])

Gen_Wd = copy.deepcopy(data_in['gen'][2:5,:])
Gen_Wd[:,8] = 25
Gen_Wd[:,3] = Gen_Wd[:,8]*0.9
Gen_Wd[:,4] = -6
Gen_Wd[:,2] = 0
Gen_Wd[0,0] = 10
Gen_Wd[1,0] = 11
Gen_Wd[2,0] = 14

Gen_Cost = copy.deepcopy(data_in['gencost'][2:5,:])
Gen_Cost[:,4:7] = 0

data_in['gen'] = np.vstack((data_in['gen'], Gen_Wd))
data_in['gencost'] = np.vstack((data_in['gencost'], Gen_Cost))
data_in['gen'][0,8] = 205
data_in['gen'][1,8] = 120
data_in['gen'][2,8] = 75
data_in['gen'][3,8] = 60
data_in['gen'][4,8] = 60

# TOPO 402 and 404
Bran_Add = copy.deepcopy(data_in['branch'][4:6,:])
Bran_Add[0,0] = 8
Bran_Add[0,1] = 14
Bran_Add[1,0] = 1
Bran_Add[1,1] = 12
# Bran_Add[2,0] = 3
# Bran_Add[2,1] = 7
# Bran_Add[1:3,2:5] = copy.deepcopy(Bran_Add[0,2:5])

# Bran_Add[0,2] = Bran_Add[0,2] - 0.03
# Bran_Add[0,3] = Bran_Add[0,3] - 0.1
# Bran_Add[0,4] = Bran_Add[0,4] - 0.024

# TOPO 403
Bran_Add = np.vstack((Bran_Add,Bran_Add[0:2,:]))
Bran_Add[2,0] = 3
Bran_Add[2,1] = 7
Bran_Add[3,0] = 1
Bran_Add[3,1] = 8

data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

PD = copy.deepcopy(data_in['bus'][:,2])
QD = copy.deepcopy(data_in['bus'][:,3])

PD_Save = np.zeros((np.size(PD,0),))
QD_Save = np.zeros((np.size(QD,0),))

PGEN_Save = np.zeros((np.size(data_in['gen'][:,1],0)))
QGEN_Save = np.zeros((np.size(data_in['gen'][:,2],0)))

VM_Save = np.zeros((np.size(data_in['bus'][:,2],0)))
VA_Save = np.zeros((np.size(data_in['bus'][:,2],0)))

WD_Save = np.zeros((3,))
#
Path = "E:\\GAN_Project\\14-bus system\\New_Dataset_Response1\\"
Topo = 0

data1 = Path + "PD_save_Topo_" + str(int(Topo)) + ".npy"
data2 = Path + "QD_save_Topo_" + str(int(Topo)) + ".npy"
data3 = Path + "P_Gen_save_Topo_" + str(int(Topo)) + ".npy"
data4 = Path + "Q_Gen_save_Topo_" + str(int(Topo)) + ".npy"

PD_Use = np.load(data1)
QD_Use = np.load(data2)

PD_Use = PD_Use[1:,:]
QD_Use = QD_Use[1:,:]

PG_Use = np.load(data3)
QG_Use = np.load(data4)

PG_Use = PG_Use[1:,:]
QG_Use = QG_Use[1:,:]

WD_P_Use = PG_Use[:,5:8]
WD_Q_Use = QG_Use[:,5:8]

num_bus = np.size(PD,0)
NUM = 3000
# NUM = np.size(PD_Use,0)
Index_Delete = np.zeros((1,))

beta_PD = 0.2
beta_QD = 0.1

pf_high = data_in['bus'][:,3]/data_in['bus'][:,2]

index = np.where(np.isnan(pf_high))
pf_high[index] = 0

PD_high = copy.deepcopy(PD) * 1.7
PD_low = copy.deepcopy(PD_high) * 0.6

# data_in['gen'][3,7] = 0.0
data_in2 = copy.deepcopy(data_in)

# Outage_Save = np.load(Path + "Outage_save_PGOUT_Topo_" + str(int(1)) + ".npy")
# Outage_Save = Outage_Save[1:,:]

a2 = data_in['gencost'][0:5,4]
a1 = data_in['gencost'][0:5,5]

# data_in['gencost'][:,4] = 0.01 + np.random.randn(19,)*0.05

A2_Save = np.zeros((5,))
A1_Save = np.zeros((5,))

A2_Load = np.load(Path + "A2_save_CostC_Topo_" + str(int(1)) + ".npy")
A1_Load = np.load(Path + "A1_save_CostC_Topo_" + str(int(1)) + ".npy")

A2_Load = A2_Load[1:,:]
A1_Load = A1_Load[1:,:]

t1 = time.time()
for i in range(NUM):
    data_in = copy.deepcopy(data_in2)
    # alpha = np.random.uniform(0, 1, (num_bus))
    # w = np.random.randn((num_bus))
    # v = np.random.randn((num_bus))
    #
    # P_sample = (alpha * PD_high + (1 - alpha) * PD_low) * (1 + beta_PD * w)
    # pf = pf_high * (1 + beta_PD * v)
    # Q_sample = P_sample * pf
    #
    # data_in['bus'][:,2] = copy.deepcopy(P_sample)
    # data_in['bus'][:,3] = copy.deepcopy(Q_sample)

    data_in['bus'][:,2] = PD_Use[i,:]
    data_in['bus'][:,3] = QD_Use[i,:]

    # V_Wd1 = Wd1_D * np.random.weibull(Wd1_K)
    # V_Wd2 = Wd2_D * np.random.weibull(Wd2_K)
    # V_Wd3 = Wd3_D * np.random.weibull(Wd3_K)
    #
    # if V_Wd1 >= 0 and V_Wd1 < V_in:
    #     P_Wd1 = 0
    # elif V_Wd1 >= V_in and V_Wd1 < V_r:
    #     P_Wd1 = Rated_Capacity * (V_Wd1 - V_in) / (V_r - V_in)
    # elif V_Wd1 >= V_r and V_Wd1 < V_out:
    #     P_Wd1 = Rated_Capacity
    # else:
    #     P_Wd1 = 0
    #
    # if V_Wd2 >= 0 and V_Wd2 < V_in:
    #     P_Wd2 = 0
    # elif V_Wd2 >= V_in and V_Wd2 < V_r:
    #     P_Wd2 = Rated_Capacity * (V_Wd2 - V_in) / (V_r - V_in)
    # elif V_Wd2 >= V_r and V_Wd2 < V_out:
    #     P_Wd2 = Rated_Capacity
    # else:
    #     P_Wd2 = 0
    #
    # if V_Wd3 >= 0 and V_Wd3 < V_in:
    #     P_Wd3 = 0
    # elif V_Wd3 >= V_in and V_Wd3 < V_r:
    #     P_Wd3 = Rated_Capacity * (V_Wd3 - V_in) / (V_r - V_in)
    # elif V_Wd3 >= V_r and V_Wd3 < V_out:
    #     P_Wd3 = Rated_Capacity
    # else:
    #     P_Wd3 = 0
    #
    # data_in['gen'][5,1] = P_Wd1
    # data_in['gen'][6,1] = P_Wd2
    # data_in['gen'][7, 1] = P_Wd3
    #
    # data_in['gen'][5,8:10] = P_Wd1
    # data_in['gen'][6, 8:10] = P_Wd2
    # data_in['gen'][7, 8:10] = P_Wd3
    #
    # data_in['gen'][5,2] = P_Wd1
    # data_in['gen'][6,2] = P_Wd2
    # data_in['gen'][7, 2] = P_Wd3
    #
    # data_in['gen'][5,3:5] = P_Wd1
    # data_in['gen'][6, 3:5] = P_Wd2
    # data_in['gen'][7, 3:5] = P_Wd3

    data_in['gen'][5,1] = WD_P_Use[i,0]
    data_in['gen'][6,1] = WD_P_Use[i,1]
    data_in['gen'][7, 1] = WD_P_Use[i,2]

    data_in['gen'][5,8:10] = WD_P_Use[i,0]
    data_in['gen'][6, 8:10] = WD_P_Use[i,1]
    data_in['gen'][7, 8:10] = WD_P_Use[i,2]

    data_in['gen'][5,2] = WD_Q_Use[i,0]
    data_in['gen'][6,2] = WD_Q_Use[i,1]
    data_in['gen'][7, 2] = WD_Q_Use[i,2]

    data_in['gen'][5,3:5] = WD_Q_Use[i,0]
    data_in['gen'][6, 3:5] = WD_Q_Use[i,1]
    data_in['gen'][7, 3:5] = WD_Q_Use[i,2]

    # PG_Out = random.randrange(2, 5)
    # data_in['gen'][PG_Out, 7] = 0.0
    # a2_Change = (1 + np.random.randn(5, ) * 0.05) * a2
    # a1_Change = (1 + np.random.randn(5, ) * 0.2) * a1

    # a2_Change = A2_Load[i,:]
    # a1_Change = A1_Load[i,:]

    # data_in['gencost'][0:5, 4] = a2_Change
    # data_in['gencost'][0:5, 5] = a1_Change

    ResRes, Ybus = runopf(data_in)
    # Outage_Save = np.vstack((Outage_Save, PG_Out))
    # A2_Save = np.vstack((A2_Save, a2_Change))
    # A1_Save = np.vstack((A1_Save, a1_Change))

    if ResRes['success'] == True:

        PD_Save = np.vstack((PD_Save, data_in['bus'][:,2]))
        QD_Save = np.vstack((QD_Save, data_in['bus'][:,3]))
        PGEN_Save = np.vstack((PGEN_Save, ResRes['gen'][:,1]))
        QGEN_Save = np.vstack((QGEN_Save, ResRes['gen'][:,2]))
        VM_Save = np.vstack((VM_Save, ResRes['bus'][:,7]))
        VA_Save = np.vstack((VA_Save, ResRes['bus'][:,8]))
    else:
        Index_Delete = np.vstack((Index_Delete, i))
t2 = time.time()
print(t2-t1)
# %%
#
Topo = 4
data1 = Path + "PD_save_CostC_Topo_" + str(int(Topo)) + ".npy"
data2 = Path + "QD_save_CostC_Topo_" + str(int(Topo)) + ".npy"
data3 = Path + "P_Gen_save_CostC_Topo_" + str(int(Topo)) + ".npy"
data4 = Path + "Q_Gen_save_CostC_Topo_" + str(int(Topo)) + ".npy"
data5 = Path + "Vol_mag_save_CostC_Topo_" + str(int(Topo)) + ".npy"
data6 = Path + "Vol_ang_save_CostC_Topo_" + str(int(Topo)) + ".npy"
data7 = Path + "A2_save_CostC_Topo_" + str(int(Topo)) + ".npy"
data8 = Path + "A1_save_CostC_Topo_" + str(int(Topo)) + ".npy"

np.save(data1, PD_Save)
np.save(data2, QD_Save)
np.save(data3, PGEN_Save)
np.save(data4, QGEN_Save)
np.save(data5, VM_Save)
np.save(data6, VA_Save)
np.save(data7, A2_Save)
np.save(data8, A1_Save)




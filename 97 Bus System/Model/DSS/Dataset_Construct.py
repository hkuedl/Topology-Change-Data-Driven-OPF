import matplotlib.pyplot as plt
import numpy as np
from pyomo.core import *
import pyomo.kernel as pyo
from pypower.api import case30,ppoption, runopf, runpf, case118, case14, case9, rundcpf, rundcopf
# from makeYbus_my import *
import copy
from makeYft import *

data_in = case14()

Gen_Cost = copy.deepcopy(data_in['gencost'][2:5,:])
Gen_Cost[:,4:7] = 0
data_in['bus'][:,2] = np.abs(data_in['bus'][:,2])
data_in['bus'][:,3] = np.abs(data_in['bus'][:,3])
data_in['branch'][:,5] = 70

data_in['gen'][0,8] = 147
data_in['gen'][1,8] = 120
data_in['gen'][2,8] = 75
data_in['gen'][3,8] = 60
data_in['gen'][4,8] = 60

Bran_Add = copy.deepcopy(data_in['branch'][4:6,:])
Bran_Add[0,0] = 8
Bran_Add[0,1] = 14
Bran_Add[1,0] = 1
Bran_Add[1,1] = 12

Bran_Add = np.vstack((Bran_Add,Bran_Add))
Bran_Add[2,0] = 3
Bran_Add[2,1] = 7
Bran_Add[3,0] = 1
Bran_Add[3,1] = 8

data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

data_branch = data_in['branch']
y = 1 / (data_branch[:, 2] + 1j * data_branch[:, 3])
bsh = data_branch[:, 4]
tap = data_branch[:, 8]
shift = data_branch[:, 9] * np.pi / 180
m = data_branch.shape[0]

Ybus = makeYft(m, y, bsh, tap, shift)
# %%
num_bus = data_in['bus'].shape[0]
num_line = data_in['branch'].shape[0]
# %%
Path_1 = "E:\\GAN_Project\\14-bus system\\GNN\\dataset\\"
A_Train = np.zeros((1, num_line, 6))
B_Train = np.zeros((1, num_bus, 2))
U_Train = np.zeros((1, num_bus, 4))
PG_Idx = [0, 1, 2, 5, 7]


for Topo in range(1, 6, 1):
    data1 = Path_1 + "PD_save_Topo_" + str(int(Topo)) + ".npy"
    data2 = Path_1 + "QD_save_Topo_" + str(int(Topo)) + ".npy"
    data3 = Path_1 + "P_Gen_save_Topo_" + str(int(Topo)) + ".npy"
    data4 = Path_1 + "Q_Gen_save_Topo_" + str(int(Topo)) + ".npy"
    data5 = Path_1 + "Vol_mag_save_Topo_" + str(int(Topo)) + ".npy"
    data6 = Path_1 + "Vol_ang_save_Topo_" + str(int(Topo)) + ".npy"
    data9 = Path_1 + "LMP_save_Topo_" + str(int(Topo)) + ".npy"

    PD_Save = np.load(data1)
    QD_Save = np.load(data2)
    PGEN_Save = np.load(data3)
    QGEN_Save = np.load(data4)
    VM_Save = np.load(data5)
    VA_Save = np.load(data6)
    LMP_Save = np.load(data9)

    if Topo == 2:
        PD_Save = PD_Save[501:557,:]
        QD_Save = QD_Save[501:557,:]
        PGEN_Save = PGEN_Save[501:557,:]
        QGEN_Save = QGEN_Save[501:557,:]
        VM_Save = VM_Save[501:557,:]
        VA_Save = VA_Save[501:557,:]
        LMP_Save = LMP_Save[501:557,:]
    else:
        PD_Save = PD_Save[1:501,:]
        QD_Save = QD_Save[1:501,:]
        PGEN_Save = PGEN_Save[1:501,0:5]
        QGEN_Save = QGEN_Save[1:501,0:5]
        VM_Save = VM_Save[1:501,:]
        VA_Save = VA_Save[1:501,:]
        LMP_Save = LMP_Save[1:501,:]

    PGEN_Total = np.zeros_like(PD_Save)
    QGEN_Total = np.zeros_like(QD_Save)
    PGEN_Save = PGEN_Save / 100
    QGEN_Save = QGEN_Save / 100

    PGEN_Total[:, PG_Idx] = PGEN_Save
    QGEN_Total[:, PG_Idx] = QGEN_Save

    num_data = PD_Save.shape[0]

    VA_Save = VA_Save * np.pi / 180
    PD_Save = PD_Save / 100
    QD_Save = QD_Save / 100

    A_Temp = np.zeros((num_data, num_line, 6))
    B_Temp = np.zeros((num_data, num_bus, 2))
    U_Temp = np.zeros((num_data, num_bus, 4))

    for i in range(num_data):
        B_Temp[i, :, 0] = np.transpose(PD_Save[i, :])
        B_Temp[i, :, 1] = np.transpose(QD_Save[i, :])

        U_Temp[i, :, 0] = np.transpose(PGEN_Total[i, :])
        U_Temp[i, :, 1] = np.transpose(QGEN_Total[i, :])
        U_Temp[i, :, 2] = np.transpose(VM_Save[i, :])
        U_Temp[i, :, 3] = np.transpose(VA_Save[i, :])

    if Topo == 1:
        data_in = case14()

        Bran_Add = copy.deepcopy(data_in['branch'][4:6, :])
        Bran_Add[0, 0] = 8
        Bran_Add[0, 1] = 14
        Bran_Add[1, 0] = 1
        Bran_Add[1, 1] = 12

        data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

        data_branch = data_in['branch']
        y = 1 / (data_branch[:, 2] + 1j * data_branch[:, 3])
        bsh = data_branch[:, 4]
        tap = data_branch[:, 8]
        shift = data_branch[:, 9] * np.pi / 180
        m = data_branch.shape[0]

        Ybus = makeYft(m, y, bsh, tap, shift)

        idx_x, idx_y = np.where(Ybus != 0)
        num_Y = idx_x.shape[0]
        for i in range(num_Y):
            A_Temp[:, i, 0] = idx_x[i]
            A_Temp[:, i, 1] = idx_y[i]
            A_Temp[:, i, 2] = np.real(Ybus[idx_x[i], idx_y[i]])
            A_Temp[:, i, 3] = np.imag(Ybus[idx_x[i], idx_y[i]])

    elif Topo == 2:
        data_in = case14()

        Bran_Add = copy.deepcopy(data_in['branch'][4:5, :])
        Bran_Add[0, 0] = 8
        Bran_Add[0, 1] = 14

        data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

        data_branch = data_in['branch']
        y = 1 / (data_branch[:, 2] + 1j * data_branch[:, 3])
        bsh = data_branch[:, 4]
        tap = data_branch[:, 8]
        shift = data_branch[:, 9] * np.pi / 180
        m = data_branch.shape[0]

        Ybus = makeYft(m, y, bsh, tap, shift)

        idx_x, idx_y = np.where(Ybus != 0)
        num_Y = idx_x.shape[0]
        for i in range(num_Y):
            A_Temp[:, i, 0] = idx_x[i]
            A_Temp[:, i, 1] = idx_y[i]
            A_Temp[:, i, 2] = np.real(Ybus[idx_x[i], idx_y[i]])
            A_Temp[:, i, 3] = np.imag(Ybus[idx_x[i], idx_y[i]])
    elif Topo == 3:
        data_in = case14()

        Bran_Add = copy.deepcopy(data_in['branch'][4:6, :])
        Bran_Add[0, 0] = 8
        Bran_Add[0, 1] = 14
        Bran_Add[1, 0] = 1
        Bran_Add[1, 1] = 12

        Bran_Add = np.vstack((Bran_Add, Bran_Add[0:1,:]))
        Bran_Add[2, 0] = 3
        Bran_Add[2, 1] = 7

        data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

        data_branch = data_in['branch']
        y = 1 / (data_branch[:, 2] + 1j * data_branch[:, 3])
        bsh = data_branch[:, 4]
        tap = data_branch[:, 8]
        shift = data_branch[:, 9] * np.pi / 180
        m = data_branch.shape[0]

        Ybus = makeYft(m, y, bsh, tap, shift)

        idx_x, idx_y = np.where(Ybus != 0)
        num_Y = idx_x.shape[0]
        for i in range(num_Y):
            A_Temp[:, i, 0] = idx_x[i]
            A_Temp[:, i, 1] = idx_y[i]
            A_Temp[:, i, 2] = np.real(Ybus[idx_x[i], idx_y[i]])
            A_Temp[:, i, 3] = np.imag(Ybus[idx_x[i], idx_y[i]])
    elif Topo == 4:
        data_in = case14()

        Bran_Add = copy.deepcopy(data_in['branch'][4:6, :])
        Bran_Add[0, 0] = 8
        Bran_Add[0, 1] = 14
        Bran_Add[1, 0] = 1
        Bran_Add[1, 1] = 12

        Bran_Add = np.vstack((Bran_Add, Bran_Add))
        Bran_Add[2, 0] = 3
        Bran_Add[2, 1] = 7
        Bran_Add[3, 0] = 1
        Bran_Add[3, 1] = 8

        data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

        data_branch = data_in['branch']
        y = 1 / (data_branch[:, 2] + 1j * data_branch[:, 3])
        bsh = data_branch[:, 4]
        tap = data_branch[:, 8]
        shift = data_branch[:, 9] * np.pi / 180
        m = data_branch.shape[0]

        Ybus = makeYft(m, y, bsh, tap, shift)

        idx_x, idx_y = np.where(Ybus != 0)
        num_Y = idx_x.shape[0]
        for i in range(num_Y):
            A_Temp[:, i, 0] = idx_x[i]
            A_Temp[:, i, 1] = idx_y[i]
            A_Temp[:, i, 2] = np.real(Ybus[idx_x[i], idx_y[i]])
            A_Temp[:, i, 3] = np.imag(Ybus[idx_x[i], idx_y[i]])
    else:
        data_in = case14()

        Bran_Add = copy.deepcopy(data_in['branch'][4:6, :])
        Bran_Add[0, 0] = 8
        Bran_Add[0, 1] = 14
        Bran_Add[1, 0] = 8
        Bran_Add[1, 1] = 9

        data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

        data_branch = data_in['branch']
        y = 1 / (data_branch[:, 2] + 1j * data_branch[:, 3])
        bsh = data_branch[:, 4]
        tap = data_branch[:, 8]
        shift = data_branch[:, 9] * np.pi / 180
        m = data_branch.shape[0]

        Ybus = makeYft(m, y, bsh, tap, shift)

        idx_x, idx_y = np.where(Ybus != 0)
        num_Y = idx_x.shape[0]
        for i in range(num_Y):
            A_Temp[:, i, 0] = idx_x[i]
            A_Temp[:, i, 1] = idx_y[i]
            A_Temp[:, i, 2] = np.real(Ybus[idx_x[i], idx_y[i]])
            A_Temp[:, i, 3] = np.imag(Ybus[idx_x[i], idx_y[i]])

    A_Train = np.vstack((A_Train, A_Temp))
    B_Train = np.vstack((B_Train, B_Temp))
    U_Train = np.vstack((U_Train, U_Temp))

A_Train = A_Train[1:, :, :]
B_Train = B_Train[1:, :, :]
U_Train = U_Train[1:, :, :]

# %%
Path_1 = "E:\\GAN_Project\\14-bus system\\GNN\\dataset\\"
A_Test = np.zeros((1, 2*num_line + num_bus, 4))
B_Test = np.zeros((1, num_bus, 2))
U_Test = np.zeros((1, num_bus, 4))
PG_Idx = [0, 1, 2, 5, 7]

for Topo in range(2, 3, 1):
    data1 = Path_1 + "PD_save_Topo_" + str(int(Topo)) + ".npy"
    data2 = Path_1 + "QD_save_Topo_" + str(int(Topo)) + ".npy"
    data3 = Path_1 + "P_Gen_save_Topo_" + str(int(Topo)) + ".npy"
    data4 = Path_1 + "Q_Gen_save_Topo_" + str(int(Topo)) + ".npy"
    data5 = Path_1 + "Vol_mag_save_Topo_" + str(int(Topo)) + ".npy"
    data6 = Path_1 + "Vol_ang_save_Topo_" + str(int(Topo)) + ".npy"
    data9 = Path_1 + "LMP_save_Topo_" + str(int(Topo)) + ".npy"

    PD_Save = np.load(data1)
    QD_Save = np.load(data2)
    PGEN_Save = np.load(data3)
    QGEN_Save = np.load(data4)
    VM_Save = np.load(data5)
    VA_Save = np.load(data6)
    LMP_Save = np.load(data9)

    PD_Save = PD_Save[701:901,:]
    QD_Save = QD_Save[701:901,:]
    PGEN_Save = PGEN_Save[701:901,:]
    QGEN_Save = QGEN_Save[701:901,:]
    VM_Save = VM_Save[701:901,:]
    VA_Save = VA_Save[701:901,:]
    LMP_Save = LMP_Save[701:901,:]

    PGEN_Total = np.zeros_like(PD_Save)
    QGEN_Total = np.zeros_like(QD_Save)
    PGEN_Save = PGEN_Save / 100
    QGEN_Save = QGEN_Save / 100

    PGEN_Total[:, PG_Idx] = PGEN_Save
    QGEN_Total[:, PG_Idx] = QGEN_Save

    num_data = PD_Save.shape[0]

    VA_Save = VA_Save * np.pi / 180
    PD_Save = PD_Save / 100
    QD_Save = QD_Save / 100

    A_Temp = np.zeros((num_data, 2*num_line + num_bus, 4))
    B_Temp = np.zeros((num_data, num_bus, 2))
    U_Temp = np.zeros((num_data, num_bus, 4))

    for i in range(num_data):
        B_Temp[i, :, 0] = np.transpose(PD_Save[i, :])
        B_Temp[i, :, 1] = np.transpose(QD_Save[i, :])

        U_Temp[i, :, 0] = np.transpose(PGEN_Total[i, :])
        U_Temp[i, :, 1] = np.transpose(QGEN_Total[i, :])
        U_Temp[i, :, 2] = np.transpose(VM_Save[i, :])
        U_Temp[i, :, 3] = np.transpose(VA_Save[i, :])

    if Topo == 1:
        data_in = case14()

        Bran_Add = copy.deepcopy(data_in['branch'][4:6, :])
        Bran_Add[0, 0] = 8
        Bran_Add[0, 1] = 14
        Bran_Add[1, 0] = 1
        Bran_Add[1, 1] = 12

        data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

        data_branch = data_in['branch']
        y = 1 / (data_branch[:, 2] + 1j * data_branch[:, 3])
        bsh = data_branch[:, 4]
        tap = data_branch[:, 8]
        shift = data_branch[:, 9] * np.pi / 180
        m = data_branch.shape[0]

        Ybus = makeYft(m, y, bsh, tap, shift)

        idx_x, idx_y = np.where(Ybus != 0)
        num_Y = idx_x.shape[0]
        for i in range(num_Y):
            A_Temp[:, i, 0] = idx_x[i]
            A_Temp[:, i, 1] = idx_y[i]
            A_Temp[:, i, 2] = np.real(Ybus[idx_x[i], idx_y[i]])
            A_Temp[:, i, 3] = np.imag(Ybus[idx_x[i], idx_y[i]])

    elif Topo == 2:
        data_in = case14()

        Bran_Add = copy.deepcopy(data_in['branch'][4:5, :])
        Bran_Add[0, 0] = 8
        Bran_Add[0, 1] = 14

        data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

        data_branch = data_in['branch']
        y = 1 / (data_branch[:, 2] + 1j * data_branch[:, 3])
        bsh = data_branch[:, 4]
        tap = data_branch[:, 8]
        shift = data_branch[:, 9] * np.pi / 180
        m = data_branch.shape[0]

        Ybus = makeYft(m, y, bsh, tap, shift)

        idx_x, idx_y = np.where(Ybus != 0)
        num_Y = idx_x.shape[0]
        for i in range(num_Y):
            A_Temp[:, i, 0] = idx_x[i]
            A_Temp[:, i, 1] = idx_y[i]
            A_Temp[:, i, 2] = np.real(Ybus[idx_x[i], idx_y[i]])
            A_Temp[:, i, 3] = np.imag(Ybus[idx_x[i], idx_y[i]])
    elif Topo == 3:
        data_in = case14()

        Bran_Add = copy.deepcopy(data_in['branch'][4:6, :])
        Bran_Add[0, 0] = 8
        Bran_Add[0, 1] = 14
        Bran_Add[1, 0] = 1
        Bran_Add[1, 1] = 12

        Bran_Add = np.vstack((Bran_Add, Bran_Add[0:1,:]))
        Bran_Add[2, 0] = 3
        Bran_Add[2, 1] = 7

        data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

        data_branch = data_in['branch']
        y = 1 / (data_branch[:, 2] + 1j * data_branch[:, 3])
        bsh = data_branch[:, 4]
        tap = data_branch[:, 8]
        shift = data_branch[:, 9] * np.pi / 180
        m = data_branch.shape[0]

        Ybus = makeYft(m, y, bsh, tap, shift)

        idx_x, idx_y = np.where(Ybus != 0)
        num_Y = idx_x.shape[0]
        for i in range(num_Y):
            A_Temp[:, i, 0] = idx_x[i]
            A_Temp[:, i, 1] = idx_y[i]
            A_Temp[:, i, 2] = np.real(Ybus[idx_x[i], idx_y[i]])
            A_Temp[:, i, 3] = np.imag(Ybus[idx_x[i], idx_y[i]])
    elif Topo == 4:
        data_in = case14()

        Bran_Add = copy.deepcopy(data_in['branch'][4:6, :])
        Bran_Add[0, 0] = 8
        Bran_Add[0, 1] = 14
        Bran_Add[1, 0] = 1
        Bran_Add[1, 1] = 12

        Bran_Add = np.vstack((Bran_Add, Bran_Add))
        Bran_Add[2, 0] = 3
        Bran_Add[2, 1] = 7
        Bran_Add[3, 0] = 1
        Bran_Add[3, 1] = 8

        data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

        data_branch = data_in['branch']
        y = 1 / (data_branch[:, 2] + 1j * data_branch[:, 3])
        bsh = data_branch[:, 4]
        tap = data_branch[:, 8]
        shift = data_branch[:, 9] * np.pi / 180
        m = data_branch.shape[0]

        Ybus = makeYft(m, y, bsh, tap, shift)

        idx_x, idx_y = np.where(Ybus != 0)
        num_Y = idx_x.shape[0]
        for i in range(num_Y):
            A_Temp[:, i, 0] = idx_x[i]
            A_Temp[:, i, 1] = idx_y[i]
            A_Temp[:, i, 2] = np.real(Ybus[idx_x[i], idx_y[i]])
            A_Temp[:, i, 3] = np.imag(Ybus[idx_x[i], idx_y[i]])
    else:
        data_in = case14()

        Bran_Add = copy.deepcopy(data_in['branch'][4:6, :])
        Bran_Add[0, 0] = 8
        Bran_Add[0, 1] = 14
        Bran_Add[1, 0] = 8
        Bran_Add[1, 1] = 9

        data_in['branch'] = np.vstack((data_in['branch'], Bran_Add))

        data_branch = data_in['branch']
        y = 1 / (data_branch[:, 2] + 1j * data_branch[:, 3])
        bsh = data_branch[:, 4]
        tap = data_branch[:, 8]
        shift = data_branch[:, 9] * np.pi / 180
        m = data_branch.shape[0]

        Ybus = makeYft(m, y, bsh, tap, shift)

        idx_x, idx_y = np.where(Ybus != 0)
        num_Y = idx_x.shape[0]
        for i in range(num_Y):
            A_Temp[:, i, 0] = idx_x[i]
            A_Temp[:, i, 1] = idx_y[i]
            A_Temp[:, i, 2] = np.real(Ybus[idx_x[i], idx_y[i]])
            A_Temp[:, i, 3] = np.imag(Ybus[idx_x[i], idx_y[i]])

    A_Test = np.vstack((A_Test, A_Temp))
    B_Test = np.vstack((B_Test, B_Temp))
    U_Test = np.vstack((U_Test, U_Temp))

A_Test = A_Test[1:, :, :]
B_Test = B_Test[1:, :, :]
U_Test = U_Test[1:, :, :]

# %%
Path_2 = "E:\\GAN_Project\\14-bus system\\Deep-Statistical-Solver-for-Distribution-System-State-Estimation\\14acopf\\"
np.save(Path_2 + "A_Train.npy", A_Train)
np.save(Path_2 + "B_Train.npy", B_Train)
np.save(Path_2 + "U_Train.npy", U_Train)

np.save(Path_2 + "A_Test.npy", A_Test)
np.save(Path_2 + "B_Test.npy", B_Test)
np.save(Path_2 + "U_Test.npy", U_Test)


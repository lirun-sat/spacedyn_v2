import Global_Value
import Set_global_value
import Get_global_value
from rpy2dc import rpy2dc

from Get_global_value import d_time
from Get_global_value import num_q
import numpy as np
import math
from eul2dc import eul2dc
from dc2rpy import dc2rpy
from f_dyn_rk2 import f_dyn_rk2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from calc_aa import calc_aa
from calc_pos import calc_pos
from j_num import j_num
from f_kin_e import f_kin_e


q = np.zeros(num_q)
qd = np.zeros(num_q)
qdd = np.zeros(num_q)

vv = np.zeros((num_q, 3))
ww = np.zeros((num_q, 3))
vd = np.zeros((num_q, 3))
wd = np.zeros((num_q, 3))

v0 = np.array([0, 0, 0])
w0 = np.array([0, 0, 0])
vd0 = np.array([0, 0, 0])
wd0 = np.array([0, 0, 0])

R0 = np.array([0, 0, 0])
Q0 = np.array([0, 0, 0])

A0 = rpy2dc(Q0[0], Q0[1], Q0[2])

print(A0)
# A0 = np.eye(3)

Fe = np.zeros((num_q, 3))
Te = np.zeros((num_q, 3))
F0 = np.array([0, 0, 0])
T0 = np.array([0, 0, 0])

tau = np.zeros(num_q)

q1tempt = []
qdtempt = []
v00tempt = []
v01tempt = []
v02tempt = []
w0tempt = []

R00_tempt = []
R01_tempt = []
R02_tempt = []

roll_1_tempt = []
pitch1_tempt = []
yaw_1_tempt = []
POS_e1_tempt = []

tau_tempt = []
A0_tempt = []

timetempt = []

# PID parameters set
desired_q = np.array([0.3, 0.2, 0.1, 0.6, 0.5, 0.4])
gain_spring = 10
gain_dumper = 10
total_steps = 40
print('total calculation : %i steps' % (total_steps / d_time))

for time in np.arange(0, total_steps, d_time):

    timetempt.append(time)
    if time == 10:
        print('1000 steps of calculation completed! Please wait~')
    elif time == 20:
        print('2000 steps of calculation completed! Please wait~')
    elif time == 30:
        print('3000 steps of calculation completed! Please wait~')
    elif time == 40:
        print('4000 steps of calculation completed! Please wait~')
    elif time == 50:
        print('5000 steps of calculation completed! Please wait~')

    tau = gain_spring * (desired_q - q) - gain_dumper * qd

    R0, A0, v0, w0, q, qd = f_dyn_rk2(R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau)

    roll_1, pitch1, yaw_1, roll_2, pitch2, yaw_2 = dc2rpy(A0)

    '''
     AA = calc_aa(A0, q)
    RR = calc_pos(R0, A0, AA, q)

    joints = j_num(0)
    POS_e1, ORI_e1 = f_kin_e(RR, AA, joints)

    POS_e1_tempt.append(POS_e1)

    # tau_tempt.append(tau)

    # joints = j_num(1)
    # POS_e2[i, :], ORI_e2[i, :, :] = f_kin_e(RR, AA, joints)
    '''

    qdtempt.append(qd[0])
    q1tempt.append(q)
    v00tempt.append(v0[0])
    v01tempt.append(v0[1])
    v02tempt.append(v0[2])

    R00_tempt.append(R0[0])
    R01_tempt.append(R0[1])
    R02_tempt.append(R0[2])

    w0tempt.append(w0[0])
    roll_1_tempt.append(roll_1)
    pitch1_tempt.append(pitch1)
    yaw_1_tempt.append(yaw_1)

    A0_tempt.append(A0[0, 0])

# plt.plot(timetempt, POS_e1_tempt, linewidth=1.0, color='red', linestyle='-.', label='POS_e1_tempt')
# plt.plot(timetempt, q1tempt, linewidth=1.0, color='red', linestyle='-.', label='q1tempt')
# plt.plot(timetempt, qdtempt, linewidth=1.0, color='black', linestyle='-.', label='qdtempt')
# plt.plot(timetempt, v00tempt, linewidth=1.0, label='v00tempt')
# plt.plot(timetempt, v01tempt, linewidth=1.0, label='v01tempt')
# plt.plot(timetempt, v02tempt, linewidth=1.0, label='v02tempt')
# plt.plot(timetempt, w0tempt, linewidth=1.0, color='blue', linestyle=':', label='w0tempt')
# plt.plot(timetempt, roll_1_tempt, linewidth=1.0, label='roll_1_tempt')
# plt.plot(timetempt, pitch1_tempt, linewidth=1.0, label='pitch1_tempt')
# plt.plot(timetempt, yaw_1_tempt, linewidth=1.0, label='yaw_1_tempt')
# plt.plot(timetempt, tau_tempt, linewidth=1.0, color='green', linestyle='-', label='tau_tempt')

plt.plot(timetempt, R00_tempt, linewidth=1.0, label='R00_tempt')
plt.plot(timetempt, R01_tempt, linewidth=1.0, label='R01_tempt')
plt.plot(timetempt, R02_tempt, linewidth=1.0, label='R02_tempt')

plt.legend(loc='upper left')

plt.grid(True)

plt.show()








'''
import test2
import test3
import test4
from test4 import ROOT

print(ROOT)

'''





'''
# use Monte Carlo to generate working space 
from calc_aa import calc_aa
from j_num import j_num
from calc_pos import calc_pos
from Get_global_value import SE
from f_kin_e import f_kin_e


A0 = np.eye(3)
R0 = np.array([0, 0, 0])

# 在[1, 10)之间均匀抽样，数组形状(1,6)
# q = np.random.uniform(-170/180 * math.pi, 170/180 * math.pi, (6,))
# print(q)
i = 0
POS_e1 = np.zeros((2000, 3))
ORI_e1 = np.zeros((2000, 3, 3))
POS_e2 = np.zeros((2000, 3))
ORI_e2 = np.zeros((2000, 3, 3))

while i < 2000:
    # A0 = np.random.rand(3, 3)

###############################################################################################################
    phi = np.random.uniform(-math.pi, math.pi)
    theta = np.random.uniform(-math.pi, math.pi)
    psi = np.random.uniform(-math.pi, math.pi)
    A0 = eul2dc(phi, theta, psi)
###############################################################################################################
    R0 = np.array([0, 0, 0])
    

    q = np.random.uniform(-170 / 180 * math.pi, 170 / 180 * math.pi, (6,))
    AA = calc_aa(A0, q)
    RR = calc_pos(R0, A0, AA, q)

    joints = j_num(0)
    POS_e1[i, :], ORI_e1[i, :, :] = f_kin_e(RR, AA, joints)

    # joints = j_num(1)
    # POS_e2[i, :], ORI_e2[i, :, :] = f_kin_e(RR, AA, joints)

    i += 1


fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(POS_e1[:, 0], POS_e1[:, 1], POS_e1[:, 2])
# ax.scatter(POS_e2[:, 0], POS_e2[:, 1], POS_e2[:, 2])



# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})


##############################################################################################################
X = POS_e1[:, 0]
Y = POS_e1[:, 1]
Z = POS_e1[:, 2]
# ax.scatter(X, Y, Z, 'b-', linewidth=4, label='curve')

null = [6]*len(Z)
ax.scatter(null, Y, Z)
ax.scatter(X, null, Z)
ax.scatter(X, Y, null)
#############################################################################################################



plt.show()
'''







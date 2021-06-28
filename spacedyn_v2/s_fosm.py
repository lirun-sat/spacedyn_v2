import numpy as np
from Get_global_value import SE
from Get_global_value import ce
from Get_global_value import num_q
from calc_aa import calc_aa
from calc_pos import calc_pos
from calc_hh import calc_hh
from r_ne import r_ne
from j_num import j_num
from calc_je import calc_je
from skew_sym import skew_sym
from Get_global_value import d_time
from f_dyn import f_dyn
from aw import aw
from rpy2qtn import rpy2qtn
from Get_global_value import Qi


def s_fosm(R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau):

    qtn_0, qtn_1, qtn_2, qtn_3 = rpy2qtn()

    vd0, wd0, qdd = f_dyn(R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau)
    x2 = np.zeros(num_q + 6)
    x2[0:3] = v0
    x2[3:6] = w0
    x2[6:num_q + 6] = qd
    sign_x2 = np.sign(x2)

    roll_1, pitch1, yaw_1, roll_2, pitch2, yaw_2 = dc2rpy(A0)
    x1 = np.zeros(num_q + 6)
    x1[0:3] = R0
    x1[3:6] =
    x1[6:num_q + 6] = q
    sign_x1 = np.sign(x1)

    s_fosm =

import numpy as np
from cross import cross


def qtn_product(P, Q):

    vec_p = np.zeros(3)
    vec_q = np.zeros(3)
    vec_p = P[1:4]
    vec_q = Q[1:4]

    scal_R = P[0] * Q[0] - np.dot(vec_p, vec_q)
    vec_R = P[0] * vec_q + Q[0] * vec_p + cross(vec_p, vec_q)

    return scal_R, vec_R[0], vec_R[1], vec_R[2]

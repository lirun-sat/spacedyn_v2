import numpy as np
from math import cos
from math import sin


def rpy2qtn(roll, pitch, yaw):

    halfYaw = yaw * 0.5
    halfPitch = pitch * 0.5
    halfRoll = roll * 0.5
    cosYaw = cos(halfYaw)
    sinYaw = sin(halfYaw)
    cosPitch = cos(halfPitch)
    sinPitch = sin(halfPitch)
    cosRoll = cos(halfRoll)
    sinRoll = sin(halfRoll)

    qtn_0 = cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw  # real part
    qtn_1 = sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw  # vector part
    qtn_2 = cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw  # vector part
    qtn_3 = cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw  # vector part

    check = qtn_0 ** 2 + qtn_1 ** 2 + qtn_2 ** 2 + qtn_3 ** 2
    if check != 1:
        print("Error Quaternion")

    return qtn_0, qtn_1, qtn_2, qtn_3

import numpy as np
import math


def dc2rpy(R):

    '''
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # x = x * 180.0/3.141592653589793
    # y = y * 180.0/3.141592653589793
    # z = z * 180.0/3.141592653589793

    return x, y, z
    '''

    pi = math.pi
    if R[2, 0] == 1:
        pitch1 = pi/2
        pitch2 = pi - pitch1
        yaw_1 = 0
        yaw_2 = yaw_1
        roll_1 = - yaw_1 + math.atan2(-R[0, 1], R[0, 2])
        roll_2 = roll_1
    elif R[2, 0] == -1:
        pitch1 = - pi / 2
        pitch2 = pi - pitch1
        yaw_1 = 0
        yaw_2 = yaw_1
        roll_1 = yaw_1 - math.atan2(R[0, 1], R[0, 2])
        roll_2 = yaw_2 - math.atan2(R[0, 1], R[0, 2])
    else:
        pitch1 = math.asin(R[2, 0])
        pitch2 = pi - pitch1
        cospitch1 = math.cos(pitch1)
        cospitch2 = math.cos(pitch2)

        roll_1 = math.atan2(-R[2, 1] / cospitch1, R[2, 2] / cospitch1)
        roll_2 = math.atan2(-R[2, 1] / cospitch2, R[2, 2] / cospitch2)

        yaw_1 = math.atan2(-R[1, 0] / cospitch1, R[0, 0] / cospitch1)
        yaw_2 = math.atan2(-R[1, 0] / cospitch2, R[0, 0] / cospitch2)

    return roll_1, pitch1, yaw_1, roll_2, pitch2, yaw_2

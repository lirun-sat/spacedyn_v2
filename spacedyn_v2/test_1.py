import numpy as np
import math
from skew_sym import skew_sym


""":arg
a = np.array([3, 4, 5])
b = skew_sym(a)
c = np.eye(3)

aa = np.array([[1, 2, 3],
               [4, 5, 6]])
"""



"""
for i in range(3, 4, 1):
    print('1+1=', 2)
    print(i)
"""



""":arg
connection = [2]
j_number = 5
while j_number != 0:
    #  connection.insert(0, j_number)
    connection = [j_number, connection]
    j_number -= 1

print(connection)
"""



"""
print(aa.T)
a_norm = np.linalg.norm(a)
d = a_norm * 3

print(a_norm)
# print(d)

e = np.true_divide(a, a_norm)
print(e)

print(a)

print(b)
print(c)

"""



""":arg
PorR = (2 == 1)
a = PorR * np.eye(3)
print(PorR)
print(a)
"""



""":arg
joints = np.array([1, 2, 3])
a = joints[-1]
print(a)
"""



""":arg
b = np.zeros((5, 3))
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(a.T)

"""




'''
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

b = np.array([[-1, -2, -3],
              [4, 5, 6],
              [7, 8, 9]])
c = np.array([1, -1, 1])

# d = np.c_[a, b]
# d = np.c_[a, c.T]
# d = np.c_[c.T, a]
# d = np.r_[a, c]  # error
# d = np.insert(a, 3, c, axis=0)
# d = np.insert(a, 3, c, axis=1)
# d = np.r_[a, b]
# d = np.r_[b, a]
# d = np.insert(a, 3, b, axis=0)
# d = np.insert(a, 3, b, axis=1)
# d = np.c_[a, b.T]
d = np.c_[a, b, c.T]
print(d)
print(type(d))

# d1 = a @ b
# print(d1)
# d2 = a @ b @ c
# print(d)

'''





'''
A0 = np.random.rand(3, 3)
print(A0)
'''




'''
import math
from eul2dc import eul2dc

phi = np.random.uniform(-math.pi, math.pi)
print(phi)

theta = np.random.uniform(-math.pi, math.pi)
print(theta)

psi = np.random.uniform(-math.pi, math.pi)
print(psi)

A0 = eul2dc(phi, theta, psi)
print(A0)
print(np.linalg.norm(A0))
print(np.linalg.inv(A0))
print(np.linalg.det(A0))
'''




'''
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()

'''



from eul2dc import eul2dc
from dc2rpy import dc2rpy
from rpy2dc import rpy2dc

pi = math.pi
a = np.array([30*3.14/180, pi/2, 120*3.14/180])
print(a)

b = rpy2dc(a[0], a[1], a[2])
print(b)

c = dc2rpy(b)
print(c)



'''
a1 = np.array([-2.61825932025646, 2.094925933333333, -1.04825932025646])
print(a1)
b1 = rpy2dc(a1[0], a1[1], a1[2])
print(b1)
c1 = dc2eul(b1)
print(c1)
'''












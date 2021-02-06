#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

res = []

# to get the first plot for tx_finger



# with open("data.txt", 'r') as f:
#     for loss in f.readlines():
#         res.append(float(loss.strip()))

# N = len(res)

# x = np.linspace(-300, 300, N)
# y = np.array(res)


# fig, ax = plt.subplots()  
# ax.plot(x, y)  
# ax.set_xlabel('horizontal shift')  
# ax.set_ylabel('loss function')  
# ax.set_title("l(p)")  
# ax.legend() 
# plt.show()

# to get the second one


with open("data_2.txt", 'r') as f:
    for loss in f.readlines():
        ligne = []
        for k in loss.split(","):
            if k != '':
                ligne.append(k)
        res.append(ligne)

for i in range(len(res)):    
    res[i].pop() 

for i in range(len(res)):
    for j in range(len(res[0])):
        res[i][j] = float(res[i][j])

ax = plt.axes(projection='3d')

x = np.linspace(-100, 100, 40)
y = np.linspace(-100, 100, 40)
x, y = np.meshgrid(x, y)

z = np.array(res)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_xlabel('horizontal shift')  
ax.set_ylabel('vertical shift')  
ax.set_zlabel('loss function') 
ax.set_title("l(p)")  
ax.legend() 


surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

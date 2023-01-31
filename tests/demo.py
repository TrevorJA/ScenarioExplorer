# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:23:12 2023

@author: tjame
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

N = 1000
XY = pd.DataFrame({'x1': np.random.rand(N),
                   'x2': np.random.rand(N),
                   'x3': np.random.rand(N),
                   'x4': np.random.rand(N),
                   'x5': np.random.rand(N)})

XY['y'] = XY['x1'] * 2*XY['x2']**2 + XY['x1']*XY['x2']**2

XY['c'] = XY.y > 0.5

plt.figure()
plt.scatter(XY[XY.c == True].x1, XY[XY.c == True].x2,  color = 'red')
plt.scatter(XY[XY.c == False].x1, XY[XY.c == False].x2,  color = 'grey')
plt.show()


X_test = pd.DataFrame({'x1': np.random.rand(N),
                   'x2': np.random.rand(N),
                   'x3': np.random.rand(N),
                   'x4': np.random.rand(N),
                   'x5': np.random.rand(N)})




model= LogisticRegression(max_iter = 1000)
model.fit(XY.loc[:, ~XY.columns.isin(['y', 'c'])], XY['c'])


# 2D plot
params = ['x1', 'x2']
exclude_columns = [params[0], params[1], 'y', 'c']

X_constant= XY.loc[:, ~XY.columns.isin(exclude_columns)]
constant_columns = X_constant.columns.values

X_variable = XY[params]

X_means = X_constant.mean()

n_samples = 10
n_constants = len(X_constant.columns)


# Make grid
x1_range = np.linspace(X_variable.min()[params[0]],
                     X_variable.max()[params[0]],
                     n_samples)

x2_range = np.linspace(X_variable.min()[params[1]],
                    X_variable.max()[params[1]],
                    n_samples)

x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)


# Make DF

grid = pd.DataFrame()

grid[params[0]] = x1_mesh.flatten()
grid[params[1]] = x2_mesh.flatten()

for i in range(n_constants):
    grid[constant_columns[i]] = np.ones(n_samples**2)*X_means[i]



grid_predictions = model.predict(grid)
Z = np.reshape(grid_predictions, np.shape(x1_mesh))
contour_levels = np.arange(0.0, 1.05,0.1)
contour_cmap = mpl.cm.get_cmap('RdBu')
dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)

fig, ax = plt.subplots()
contourset = ax.contourf(x1_mesh, x2_mesh, Z, contour_levels, cmap=contour_cmap)
#ax.scatter(XY[XY.c == True].x1, XY[XY.c == True].x2,  color = 'red')
ax.scatter(XY.x1, XY.x2, c= XY.c.values*1, edgecolor='none', cmap = dot_cmap, alpha = 0.5)

ax.set_xlim(np.min(x1_mesh),np.max(x1_mesh))
ax.set_ylim(np.min(x2_mesh),np.max(x2_mesh))
ax.set_xlabel(params[0], fontsize=24)
ax.set_ylabel(params[1], fontsize=24)
ax.tick_params(axis='both',labelsize=18)

fig.subplots_adjust(wspace=0.3,hspace=0.3,right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(contourset, cax=cbar_ax)
cbar_ax.set_ylabel('Probability of Success',fontsize=20)
#yticklabels = cbar.ax.get_yticklabels()
#cbar.ax.set_yticklabels(yticklabels,fontsize=18)
fig.set_size_inches([14.5,8])
#fig.savefig('Fig1.png')
plt.show()


#%% BT

model = GradientBoostingClassifier(max_depth = 10)
model.fit(XY.loc[:, ~XY.columns.isin(['y', 'c'])], XY['c'])

grid_predictions = model.predict(grid)
Z = np.reshape(grid_predictions, np.shape(x1_mesh))
contour_levels = np.arange(0.0, 1.05,0.1)
contour_cmap = mpl.cm.get_cmap('RdBu')
dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)

fig, ax = plt.subplots()
contourset = ax.contourf(x1_mesh, x2_mesh, Z, contour_levels, cmap=contour_cmap)
#ax.scatter(XY[XY.c == True].x1, XY[XY.c == True].x2,  color = 'red')
ax.scatter(XY.x1, XY.x2, c= XY.c.values*1, edgecolor='none', cmap = dot_cmap, alpha = 0.5)

ax.set_xlim(np.min(x1_mesh),np.max(x1_mesh))
ax.set_ylim(np.min(x2_mesh),np.max(x2_mesh))
ax.set_xlabel(params[0], fontsize=24)
ax.set_ylabel(params[1], fontsize=24)
ax.tick_params(axis='both',labelsize=18)

fig.subplots_adjust(wspace=0.3,hspace=0.3,right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(contourset, cax=cbar_ax)
cbar_ax.set_ylabel('Probability of Success',fontsize=20)
#yticklabels = cbar.ax.get_yticklabels()
#cbar.ax.set_yticklabels(yticklabels,fontsize=18)
fig.set_size_inches([14.5,8])
#fig.savefig('Fig1.png')
plt.show()

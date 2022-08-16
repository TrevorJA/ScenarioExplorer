"""
Trevor Amestoy

Contains plotting functions for LogisticRegression.


Sourced from:
https://waterprogramming.wordpress.com/2018/05/04/logistic-regression-for-scenario-discovery/

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from logistic_regression_functions import normalize_columns, fit_logistic

from utils import normalize_columns, binary_performance_mask

################################################################################

def construct_contour_map(model, ax, constant, contour_cmap, dot_cmap, levels, xgrid, ygrid, \
    xvar, yvar, base):

    dta = normalize_columns(model.data)

    result = model.run()

    dta['Intercept'] = np.ones(np.shape(dta)[0])
    dta['Success'] = binary_performance_mask(model)

    # find probability of success for x=xgrid, y=ygrid
    X, Y = np.meshgrid(xgrid, ygrid)
    x = X.flatten()
    y = Y.flatten()


    # GENERALIZE THIS
    if constant == 'x3': # 3rd predictor held constant at base value
        grid = np.column_stack([x, y, np.ones(len(x))*base[2], np.ones(len(x))])

    elif constant == 'x2': # 2nd predictor held constant at base value
        grid = np.column_stack([x, np.ones(len(x))*base[1], y, np.ones(len(x))])

    else: # 1st predictor held constant at base value
        grid = np.column_stack([np.ones(len(x))*base[0], x, y, np.ones(len(x))])

    #model.grid = grid
    #model.x = x
    #model.y = y
    #model.dta = dta
    #model.z = z
    #model.Z = Z

    z = result.predict(grid)
    Z = np.reshape(z, np.shape(X))

    contourset = ax.contourf(X, Y, Z, levels, cmap=contour_cmap)
    ax.scatter(dta[xvar].values, dta[yvar].values, c=dta['Success'].values, edgecolor='none', cmap=dot_cmap, alpha = 0.2)
    ax.set_xlim(np.min(X),np.max(X))
    ax.set_ylim(np.min(Y),np.max(Y))
    ax.set_xlabel(xvar,fontsize=24)
    ax.set_ylabel(yvar,fontsize=24)
    ax.tick_params(axis='both',labelsize=18)

    return contourset


################################################################################

def plot_single_contour(model, x, y):

    # Color map for dots representing success (light blue) and fails (dark red)
    dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)

    # Define color map for probability contours
    contour_cmap = mpl.cm.get_cmap('RdBu')

    # Define probability contours
    contour_levels = np.arange(0.0, 1.05,0.1)

    # Define grids for each predictor (they are normalized)
    xgrid = np.arange(-0.1,1.1,0.01)
    ygrid = np.arange(-0.1,1.1,0.01)

    # define base values of 3 predictors
    base = np.mean(normalize_columns(model.data)).values
    print(f'Base: {base}')

    # Drop intercept and success columns
    param_labs = model.data.columns

    # Determine how many subplots to make
    n_params = int(len(model.data.columns))

    # Set up plot
    fig, ax = plt.subplots()

    constant_col = [x for x in param_labs if x not in [x, y]]
    constant_col = 'x1'
    print(f'Constant: {constant_col}')

        # plot contour map when 3rd predictor ('x3') is held constant
    contourset = construct_contour_map(model, ax, constant_col, contour_cmap, dot_cmap, contour_levels, xgrid, ygrid, \
        x, y, base)

    # Finish the plot
    fig.subplots_adjust(wspace=0.3,hspace=0.3,right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(contourset, cax=cbar_ax)
    cbar_ax.set_ylabel('Probability of Success',fontsize=20)
    #yticklabels = cbar.ax.get_yticklabels()
    #cbar.ax.set_yticklabels(yticklabels,fontsize=18)
    fig.set_size_inches([14.5,8])
    fig.savefig('Fig1.png')
    plt.show()
    fig.clf()
    return

################################################################################


def plot_many_contours(model):
    """WIP"""

    # Color map for dots representing success (light blue) and fails (dark red)
    dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)

    # Define color map for probability contours
    contour_cmap = mpl.cm.get_cmap('RdBu')

    # Define probability contours
    contour_levels = np.arange(0.0, 1.05,0.1)

    # Define grids for each predictor (they are normalized)
    xgrid = np.arange(-0.1,1.1,0.01)
    ygrid = np.arange(-0.1,1.1,0.01)

    # define base values of 3 predictors
    base = [0.5, 0.5, 0.5]

    # Drop intercept and success columns
    param_data = model.data.drop(['Intercept', 'Success'], axis = 1)


    # Determine how many subplots to make
    n_params = int(len(model.data.columns))
    n_row = n_params / 3
    n_col = (n_params - 1) / n_row

    # Set up plot
    fig, axs = plt.subplots(n_row, n_col)

    for i, constant_col in enumerate(param_data.columns):

        ax = axs[i]

        # Pull param names
        x = param_data.columns[i]
        y = param_data.columns[i+1]
        constant_col = param_data.columns[i+2]

        # plot contour map when 3rd predictor ('x3') is held constant
        construct_contour_map(model, ax, constant_col, contour_cmap, dot_cmap, contour_levels, xgrid, ygrid, \
        x, y, base)



    # Finish the plot
    fig.subplots_adjust(wspace=0.3,hspace=0.3,right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(contourset, cax=cbar_ax)
    cbar_ax.set_ylabel('Probability of Success',fontsize=20)
    yticklabels = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(yticklabels,fontsize=18)
    fig.set_size_inches([14.5,8])
    fig.savefig('Fig1.png')
    plt.show()
    fig.clf()
    return

"""
Trevor Amestoy
Cornell University

UNDER DEVELOPMENT: DO NOT USE
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl



################################################################################

def construct_contour_map(model, ax, constant, contour_cmap, dot_cmap, levels, xgrid, ygrid, \
    variable_params, base):

    xvar = variable_params[0]
    yvar = variable_params[1]

    dta = normalize_columns(model.data)

    all_params = dta.columns.to_list()

    result = model.fit_logistic(subset_predictors= True, subset = variable_params)

    dta['Intercept'] = np.ones(np.shape(dta)[0])
    dta['Success'] = binary_performance_mask(model)

    # find probability of success for x=xgrid, y=ygrid
    X, Y = np.meshgrid(xgrid, ygrid)
    x = X.flatten()
    y = Y.flatten()

    variable_param_indeces = [all_params.index(variable_params[i]) for i in range(len(variable_params))]

    # Create a grid of 1s with n-dimensions corresponding to n-parameters (add 1 more for intercept)
    grid = [np.ones(len(x)) for _ in range(len(variable_params) + 1)]

    for i in range(len(variable_params)):

        if i == variable_param_indeces[0]:
            grid[i] = x

        elif i == variable_param_indeces[1]:
            grid[i] = y

        else:
            grid[i] = grid[i] * base[i]

    grid = np.column_stack(grid)

    model.grid = grid
    #model.x = x
    #model.y = y
    #model.dta = dta

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

def plot_single_contour(model, variable_params):

    assert (len(variable_params) == 2), 'variable_params must contain only two parameter names.'

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

    # Define the constant parameters as those not in the variable pair
    constant_params = [x for x in param_labs if x not in variable_params]
    print(f'Constants: {constant_params}')

    # plot contour map when 3rd predictor ('x3') is held constant
    contourset = construct_contour_map(model, ax, constant_params, contour_cmap, dot_cmap, contour_levels, xgrid, ygrid, \
        variable_params, base)

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


def plot_area_SOS_over_range(model, variable_params, threshold_range):

    assert (len(variable_params) == 2), 'variable_params must contain only two parameter names.'

    # Color map for dots representing success (light blue) and fails (dark red)
    dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)

    # Define color map for probability contours
    contour_cmap = mpl.cm.get_cmap('RdBu')

    # Define probability contours
    contour_levels = np.arange(0.0, 1.0,0.1)

    # Define grids for each predictor (they are normalized)
    xgrid = np.arange(0,1,0.01)
    ygrid = np.arange(0,1,0.01)

    # define base values of 3 predictors
    base = np.mean(normalize_columns(model.data)).values
    print(f'Base: {base}')

    # Drop intercept and success columns
    param_labs = model.data.columns

    # Determine how many subplots to make
    n_params = int(len(model.data.columns))

    # Define the constant parameters as those not in the variable pair
    constant_params = [x for x in param_labs if x not in variable_params]
    print(f'Constants: {constant_params}')

    # Calculate the contour set
    xvar = variable_params[0]
    yvar = variable_params[1]

    dta = normalize_columns(model.data)
    all_params = dta.columns.to_list()

    dta['Intercept'] = np.ones(np.shape(dta)[0])
    variable_param_indeces = [all_params.index(variable_params[i]) for i in range(len(variable_params))]

    # A list of area sizes
    a = []

    for n in range(threshold_range[0], threshold_range[1]):
        model.threshold = n

        result = model.run()

        dta['Success'] = binary_performance_mask(model)

        # find probability of success for x=xgrid, y=ygrid
        X, Y = np.meshgrid(xgrid, ygrid)
        x = X.flatten()
        y = Y.flatten()

        # Create a grid of 1s with n-dimensions corresponding to n-parameters (add 1 more for intercept)
        grid = [np.ones(len(x)) for _ in range(len(all_params) + 1)]

        for i in range(len(all_params)):

            if i == variable_param_indeces[0]:
                grid[i] = x
            elif i == variable_param_indeces[1]:
                grid[i] = y
            else:
                grid[i] = grid[i] * base[i]

        grid = np.column_stack(grid)

        model.grid = grid

        z = result.predict(grid)
        Z = np.reshape(z, np.shape(X))

        cs = plt.contour(X, Y, Z, contour_levels)

        model.cs = cs

        # Get the SOS contour
        contour = cs.collections[-1]
        vertices = contour.get_paths()[0].vertices

        # Calculate the area of the contour
        a.append(calculate_linear_contour_area(vertices))
        model.areas = a


    # Plot results
    fig, ax = plt.subplots(dpi = 150)
    ax.plot(range(threshold_range[0], threshold_range[1]), a, label = '$Area_{SOS}$', color = 'green')
    ax.set_title('Influence of threshold value on safe operating space')
    ax.set_xlabel('Threshold value')
    ax.set_ylabel('Relative area of SOS')
    plt.legend()
    fig.set_size_inches([7,5])
    fig.savefig('area_of_SOS.png')
    plt.show()
    fig.clf()
    return










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

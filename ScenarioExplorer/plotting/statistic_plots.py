"""
Trevor Amestoy
Cornell University

UNDER DEVELOPMENT: DO NOT USE
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ScenarioExplorer.utils.classification

def plot_ranges(var1_vals, runs, variable_name, compare = True, var2_vals = 0.0, second_var_name = None):

    min_vals = np.min(var1_vals, axis = 0)
    max_vals = np.max(var1_vals, axis = 0)

    if min_vals[-1] == 0:
        min_vals = min_vals[0:-1]
        max_vals = max_vals[0:-1]

    x = range(len(min_vals))
    if len(min_vals) > 21:
        x_lab = 'Month'
    else:
        x_lab = 'Year'

    plt.fill_between(x, min_vals, max_vals, color = 'green', alpha = 0.1, label = runs[0])
    plt.plot(x, min_vals, color = 'black')
    plt.plot(x, max_vals, color = 'black')

    if compare:
        min_vals = np.min(var2_vals, axis = 0)
        max_vals = np.max(var2_vals, axis = 0)

        if min_vals[-1] == 0:
            min_vals = min_vals[0:-1]
            max_vals = max_vals[0:-1]
        x = range(len(min_vals))
        plt.fill_between(x, min_vals, max_vals, color = 'blue', alpha = 0.1, label = runs[1])
        plt.plot(x, min_vals, color = 'black')
        plt.plot(x, max_vals, color = 'black')


    plt.title(f'Range of {variable_name} across realizations\n Run {runs}')
    plt.ylabel(f'{variable_name}')
    plt.xlabel(x_lab)
    plt.legend()
    plt.show()
    return None






def plot_var_combinations(var1, var2, performance, var1_name, var2_name):
    if var1.shape != var2.shape:
        print('Variable sizes dont match.')
        return
    if var1.shape != performance.shape:
        print('Variable matrix shape does not match performance metric shape.')
        return

    if len(var1.shape) > 1:
        var1 = var1.flatten()
        var2 = var2.flatten()
        performance = performance.flatten()

    # Create a flat matrix of month values
    t = np.tile(np.arange(0,239), (1000,1)).flatten()

    # Generate success mask
    fail_x, x_fail_mask = mask_scenarios(var1, performance) # mask = True means shortfall
    fail_y, y_fail_mask = mask_scenarios(var2, performance)

    success_x = var1[x_fail_mask == False]
    success_y = var2[y_fail_mask == False]

    success_t = t[x_fail_mask == False]
    fail_t = t[x_fail_mask == True]

    print(f'len fail x = {len(fail_x)}')
    print(f'len fail y = {len(fail_y)}')
    print(f'len fail t = {len(fail_t)}')

    # Plot
    fig = plt.figure(figsize=(7, 7), dpi = 275)
    ax = fig.add_subplot(projection='3d')

    ax.scatter(fail_x, fail_t, fail_y, edgecolors = 'r', label = 'Shortfall', alpha = 0.1, marker = 'o', s = 5, facecolors = 'none')
    ax.scatter(success_x, success_t, success_y, edgecolors = 'g', label = 'No shortfalls', alpha = 0.1, marker = 'o', s = 5, facecolors = 'none')
    ax.legend(bbox_to_anchor=[1.1, 0.0])
    ax.set_xlabel(var1_name)
    ax.set_ylabel('Time (months)')
    ax.set_zlabel(var2_name)
    ax.set_xlim([min(var1), max(var1)])
    ax.set_ylim([0, 239])
    ax.set_zlim([min(var2), max(var2)])
    ax.view_init(8, -35)
    plt.show()
    plt.close('all')

    return None



################################################################################

def plot_histogram(data):
    hist, bins = np.histogram(data)
    fig = plt.figure()
    plt.hist(data.flatten(), bins)
    plt.show()
    return None

################################################################################

def animate_scatter(i, time_scatter, plot_title, var1, var2):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct','Nov','Dec']
    month_vect = np.tile(months, (20, 1)).flatten()
    year = 2020 + int(np.floor(i / 12))
    month = month_vect[i]
    time_scatter.set_offsets(np.c_[var1[:, i], var2[:,i]])
    plot_title.set_text(f'{month} {year}')
    print(f'Month {i}')


################################################################################

def animate_var_combinations(var1, var2, performance, var1_name, var2_name):
    if var1.shape != var2.shape:
        print('Variable sizes dont match.')
        return
    if var1.shape != performance.shape:
        print('Variable matrix shape does not match performance metric shape.')
        return

    if len(var1.shape) > 1:
        var1_flat = var1.flatten()
        var2_flat = var2.flatten()
        performance_flat = performance.flatten()

    # Create a flat matrix of month values
    t = np.tile(np.arange(0,239), (1000,1)).flatten()

    # Generate success mask
    fail_x, x_fail_mask = mask_scenarios(var1_flat, performance_flat) # mask = True means shortfall
    fail_y, y_fail_mask = mask_scenarios(var2_flat, performance_flat)

    success_x = var1_flat[x_fail_mask == False]
    success_y = var2_flat[y_fail_mask == False]

    success_t = t[x_fail_mask == False]
    fail_t = t[x_fail_mask == True]

    print(f'len fail x = {len(fail_x)}')
    print(f'len fail y = {len(fail_y)}')
    print(f'len fail t = {len(fail_t)}')

    # Plot
    fig,ax = plt.subplots(figsize=(7, 7), dpi = 275)

    #ax.legend(bbox_to_anchor=[1.1, 0.0])
    ax.set_xlabel(var1_name)
    ax.set_ylabel(var2_name)

    ax.set_xlim([min(var1_flat), max(var1_flat)])
    ax.set_ylim([min(var2_flat), max(var2_flat)])

    static_scatter2 = ax.scatter(success_x, success_y, edgecolors = 'grey', label = 'No shortfalls', alpha = 0.1, marker = 'o', s = 5, facecolors = 'none')
    static_scatter1 = ax.scatter(fail_x, fail_y, edgecolors = 'r', label = 'Shortfall', alpha = 0.1, marker = 'o', s = 5, facecolors = 'none')

    plot_title = ax.set_title(f'{0} {2020}')
    time_scatter = ax.scatter(var1[:,0], var2[:,0], edgecolors = 'b', marker = 'o', s = 10, facecolors = 'none')

    animation = FuncAnimation(fig, animate_scatter, fargs = (time_scatter, plot_title, var1, var2), interval = 600, frames = np.arange(238), repeat = False)
    print('Saving .gif...')
    animation.save(f'./monthly_SA_{var1_name}_{var2_name}.gif', writer = 'pillow')

    plt.show()
    plt.close('all')

    return None

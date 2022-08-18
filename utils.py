"""
Trevor Amestoy

Contains basic utility functions used in scneario discovery.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from scipy import stats

################################################################################

def binary_performance_mask(model):
    """
    model : ScenarioExplorer
    col : int
    """
    if model.threshold_type == '>':
        mask = model.performance > model.threshold

    elif model.threshold_type == '<':
        mask = model.performance < model.threshold

    else:
        raise Exception('Invalid threshold type.  Options include [>, <].')

    return 1 * (mask == False)


################################################################################

def normalize_columns(df):
    return (df-df.min())/(df.max()-df.min())



################################################################################

def mask_all_inputs(model):
    """
    model : ScenarioExplorer
    """


    n_sows = model.inputs.shape[0]
    n_params = model.inputs.shape[1]

    if model.threshold_type == '>':
        for n in range(n_params):
            data = np.ma.masked_where(model.performance > model.threshold, model.inputs[:, n])

    elif model.threshold_type == '<':
        for n in range(n_params):
            data = np.ma.masked_where(model.performance < model.threshold, model.inputs[:, n])

    else:
        raise Exception('Invalid threshold type.  Options include [>, <].')

    #masked_var = data[data.mask == True]

    return data


################################################################################


def calculate_contour_area(vs):
    """
    Source: https://stackoverflow.com/questions/22678990/how-can-i-calculate-the
            -area-within-a-contour-in-python-using-the-matplotlib#:~:text=Using%
            20the%20vertices%20you%20can,area%20of%20the%20enclosed%20region.
    """
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return a


################################################################################


################################################################################


################################################################################


################################################################################

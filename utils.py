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


def calculate_linear_contour_area(vs):
    """
    """
    x0,y0 = vs[0]
    xf, yf =vs[-1]

    if xf > x0:
        A = (x0*yf) + 0.5 * (yf * (xf - x0))
    else:
        A = (xf*yf) + 0.5 * (yf * (x0 - xf))

    return A


################################################################################


################################################################################


################################################################################


################################################################################

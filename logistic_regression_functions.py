"""
Trevor Amestoy

Adapted from code by Julie Quinn, available here:
https://waterprogramming.wordpress.com/2018/05/04/logistic-regression-for-scenario-discovery/

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from scipy import stats

################################################################################

def normalize_columns(df):
    return (df-df.min())/(df.max()-df.min())


################################################################################


def fit_logistic(df, subset_predictors = False, subset = None, normalize = True):
    """
    Parameters:
    ----------
    df : DataFrame
        Containing n columns of covariates, and a final column of binary
        performance with failure = 0 and sucess = 1.
    select_predictors : list of strings
        A list of the predictor column names to be used in the regression.

    Returns:
    -------
    fit_model
    """

    df = normalize_columns(df)

    # Reset column of intercepts
    df['Intercept'] = 1

    if subset_predictors:
        # Get a list of columns to use as predictors (and success col)
        cols = subset

    else:
        cols = df.columns.tolist()[:-1]

    # Fit regression
    logit = sm.Logit(df['Success'], df[cols])
    result = logit.fit()

    return result



def rank_significance(df):
    """
    Calculates the McFadden psuedo-R2 value of each predictor individually and
    ranks the predictors in order of significance.

    Parameters:
    -----------
    df : DataFrame
        Containing n columns of covariates, and a final column of binary
        performance with failure = 0 and sucess = 1.
    predictors : list of strings
        A list of the predictor column names to be used in the regression.

    Returns:
    --------
    rank_covariates : DataFrame
        A dataframe of all covariates, ranked by psuedo-R2.
    """

    # Count the number of total predictors (-1 for success column)
    n = len(df.columns) - 2

    # Initialize storage
    predictor_names = []
    pseudo_R2 = []

    for i in range(n):
        single_predictor = df.columns.tolist()[i:i+1]
        result = fit_logistic(df, subset_predictors = True, subset = single_predictor)

        predictor_names.append(single_predictor[0])
        pseudo_R2.append(result.prsquared)

    # Combine data into a DataFrame
    outputs = pd.DataFrame({'Predictor' : predictor_names, 'Psuedo-R2' : pseudo_R2})

    # Rank based upon the pseudo_R2
    ranked_predictors = outputs.sort_values('Psuedo-R2', ascending = False)
    return ranked_predictors

"""
Trevor Amestoy
Cornell University

Functions:

UNDER DEVELOPMENT: DO NOT USE
"""

import numpy as np
import pandas as pd


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

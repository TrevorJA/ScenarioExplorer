"""
Author: Trevor Amestoy
Contact: tja73@cornell.edu

"""

# Core imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Package imports
from ..utils import check_kwargs

class LogisticRegression():
    """
    Algorithm class used for logistic regression classification.

    Parameters
    ----------

    Methods
    -------
    fit_logistic

    Attributes
    ----------
    """
    def __init__(self, uncertainties, **kwargs):
        """
        Description of logistic.

        Parameters
        ----------
        pass

        Attributes
        ----------
        pass
        """
        

    def fit(self, **kwargs):
        """
        Description.

        Parameters
        ----------
        pass

        Attributes
        ----------
        pass
        """"

        # Noramlize columns

        # Reset column of intercepts
        df['Intercept'] = 1

        if subset_predictors:
            # Get a list of columns to use as predictors (and success col)
            print('Using a subset of predictors.')
            cols = subset.append('Intercept')
        else:
            cols = df.columns.tolist()

        # Fit regression
        logit = sm.Logit(df['Success'], df[cols])
        result = logit.fit()
        return result

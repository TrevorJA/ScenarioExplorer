"""
Trevor Amestoy
Cornell University

Combines various scenario discovery functions.

"""

# Core libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy import stats

# External functions
from logistic_regression_functions import rank_significance
from logistic_regression_plots import plot_many_contours, plot_single_contour

from utils import binary_performance_mask, normalize_columns

################################################################################

class LogisticRegression:

    def __init__(self,
                inputs, performance,
                threshold = None, threshold_type = '>'):

        # Store values
        self.inputs = inputs
        self.performance = performance
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.param_names = param_names


        # Envoke the parent class (if expanding)
    def fit_logistic(self, subset_predictors = False, subset = None, normalize = True):
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

        df = normalize_columns(self.data)

        # Add a column of intercepts
        df['Intercept'] = np.ones(np.shape(self.data)[0])

        # Apply mask to performance
        df['Success'] = binary_performance_mask(self)

        if subset_predictors:
            # Get a list of columns to use as predictors (and success col)
            cols = subset
            print(f'Using cols: {cols}')

        else:
            cols = df.drop('Success', axis = 1).columns.tolist()
            print(f'Using all cols: {cols}')


        # Fit regression
        logit = sm.Logit(df['Success'], df[cols])
        result = logit.fit()

        return result

    def run(self):

        # Make a dataframe combing inputs and performance
        self.data = pd.DataFrame(self.inputs)

        # Fit the model
        self.linear_model = self.fit_logistic()

        return self.linear_model


    def rank_inputs(self):
        self.ranks = rank_significance(self.data)
        return self.ranks


    def plot_parameter_contour_map(self, variable_params):
        """
        Produces a scatter plot of success/fail scenarios, overlayed on a
        contour map showing the probability of sucess calculated from a
        best-fit logistic regression.

        All parameters are used in the logistic regression.

        Only the variable_params are considered in the 2D plot; other params
        are constant at their base (mean) value.
        """
        plot_single_contour(self, variable_params)
        return


    def plot_all_contours(self):
        #WIP
        return



################################################################################

class PRIM:

    def algorithm_run(self):

        self.results = 0

        # Store a bool indicating successful execution
        self.complete = True
        return self.results


    def count_coi(self, indices):
        """
        Given a set of indices on y, count the number of cases of interest in the set.
        """

        y_subset = self.y[indices]
        coi = y_subset[y_subset == True].shape[0]
        return coi


################################################################################


class CART:
    def __init__(self):
        pass


################################################################################

class BoostedTrees:
    def __init__(self):
        pass


################################################################################


class ScenarioExplorer:

    def __init__(self,
                inputs, performance, param_names,
                method = 'PRIM',
                threshold = None, threshold_type = '>',
                peel_alpha = 0.05, paste_alpha = 0.05):

        """
        # Convert to np matrix
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_records(index = False)
        else:
            inputs = pd.DataFrame(inputs).to_records(index = False)


        if isinstance(performance, pd.DataFrame):
            y = performance.to_numpy()
        else:
            y = np.asarray(performance)


        Error: tuple index out of range
        if len(param_names) != inputs.shape[1]:
            raise Exception('The number of inputs (columns of input matrix) and length of param_names must be equal.')
        """

        # Verify that they are the correct dimensions
        #assert len(inputs) == len(inputs), f"Input datas must be the same size, but have sizes {len(inputs)} and {len(inputs)})"

        #assert len(y.shape) < 1, f'Performance must be a 1-d array, but has shape {y.shape}.'

        # Store values
        self.inputs = inputs
        self.performance = performance
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.peel_alpha = peel_alpha
        self.paste_alpha = paste_alpha
        self.param_names = param_names

    def mask_performance(self):
        self.success = binary_performance_mask(self)
        return self.success

    def assign_subclass(self):
        if self.method == "PRIM":
            self.algorithm = self.PRIM()

        elif self.method == "CART":
            self.algorithm = self.CART()

        elif self.method == "BoostedTrees":
            self.algorithm = self.BoostedTrees()
        return


    def run(self):
        assign_subclass(self)
        self.algorithm.algorithm_run()

    def plot(self):
        assert self.complete == True


################################################################################

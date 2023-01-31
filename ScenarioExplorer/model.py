
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


from ScenarioExplorer.algorithms.LogisticRegression import LogisticRegressionClassifier
from ScenarioExplorer.algorithms.BoostedTrees import BoostedTreeClassifier

class ScenarioExplorer:
    """
    Description.
    """

    def __init__(self, *args, **kwargs):

        # Assertions

        self.XY = args[0]

        # Sort kwargs
        self.fail_threshold = kwargs.get('fail_threshold', 0)
        self.fail_criteria = kwargs.get('fail_criteria', '>')
        self.method = kwargs.get('method', 'logistic')

        # Useful tags
        self._trained = False

    def classify(self):

        if self.fail_criteria == '>':
            self.XY["classification"] = self.XY.performance > self.fail_threshold
        elif self.fail_criteria == '>=':
            self.XY["classification"] = self.XY.performance >= self.fail_threshold
        elif self.fail_criteria == '<':
            self.XY["classification"] = self.XY.performance <= self.fail_threshold
        elif self.fail_criteria == '<=':
            self.XY["classification"] = self.XY.performance < self.fail_threshold
        else:
            print(f'{self.fail_criteria} is not a valid criteria.\n\
                    Options (entered as strings) are: <, <=, >, >=')
        return


    def train(self, **kwargs):

        # Classify as success or failure
        self.classify()

        if self.method == 'logistic':
            self.model = LogisticRegressionClassifier()
        elif self.method == 'boosted-trees':
            self.model = BoostedTreeClassifier()

        self.model.train(self.XY.loc[:, ~self.XY.columns.isin(['performance', 'classification'])],
                        self.XY['classification'])
        self._trained = True
        return


    def predict(self, X_predict):
        if self._trained:
            return self.model.predict(X_predict)
        else:
            print('Training model...')
            self.train()
            print('Making prediction...')
            return self.model.predict(X_predict)


    def plot_contour(self, vars, save_figure = False):
        n_samples = 10

        assert(len(vars) == 2), 'Input vars must be length 2.'
        # Assert vars are valid column names

        # Train the model if not done already
        if not self._trained:
            self.train()

        exclude_columns = [vars[0], vars[1], 'performance', 'classification']

        x_fixed = self.XY.loc[:, ~self.XY.columns.isin(exclude_columns)]
        fixed_vars = x_fixed.columns.values

        x_variable = self.XY[vars]

        x_fixed_means = x_fixed.mean()
        n_fixed = len(x_fixed_means)

        # Make a grid
        x1_range = np.linspace(x_variable.min()[vars[0]],
                            x_variable.max()[vars[0]],
                            n_samples)

        x2_range = np.linspace(x_variable.min()[vars[1]],
                            x_variable.max()[vars[1]],
                            n_samples)

        x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

        # Convert grid to DataFrame
        grid = pd.DataFrame()
        grid[vars[0]] = x1_mesh.flatten()
        grid[vars[1]] = x2_mesh.flatten()
        for i in range(n_fixed):
            grid[fixed_vars[i]] = np.ones(n_samples**2)*x_fixed_means[i]

        self._grid = grid
        # Re-order
        input_columns = self.XY.loc[:,
                                    ~self.XY.columns.isin(['performance',
                                                            'classification'])].columns
        grid = grid[input_columns]

        # Make predictions across the grid
        predictions = self.model.predict(grid)
        predictions = np.reshape(predictions, np.shape(x1_mesh))


        ### Plotting
        ## Colors
        if self.method == 'logistic':
            contour_levels = np.arange(0.0, 1.05,0.1)
        elif self.method == 'boosted-trees':
            contour_levels = [0.0, 0.5, 1.0]
            
        contour_cmap = mpl.cm.get_cmap('RdBu')
        dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)

        fig, ax = plt.subplots()
        contourset = ax.contourf(x1_mesh,
                                x2_mesh,
                                predictions,
                                contour_levels,
                                cmap=contour_cmap)

        ax.scatter(self.XY[vars[0]],
                    self.XY[vars[1]],
                    c= self.XY.classification.values*1,
                    edgecolor='none', cmap = dot_cmap, alpha = 0.5)


        # Aesthetics
        ax.set_xlim(np.min(x1_mesh),np.max(x1_mesh))
        ax.set_ylim(np.min(x2_mesh),np.max(x2_mesh))
        ax.set_xlabel(vars[0], fontsize=24)
        ax.set_ylabel(vars[1], fontsize=24)
        ax.tick_params(axis='both',labelsize=18)
        fig.subplots_adjust(wspace=0.3,hspace=0.3,right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(contourset, cax=cbar_ax)
        cbar_ax.set_ylabel('Probability of Success',fontsize=20)
        fig.set_size_inches([14.5,8])

        if save_figure:
            fig.savefig(f'Predicted_success_{self.method}_{vars[0]}_and_{vars[1]}.png')
        return plt

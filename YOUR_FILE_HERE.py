
import pandas as pd
import numpy as np

import ScenarioExplorer

### Create demo data
# Number of samples
N = 1000

# Random variable values
XY = pd.DataFrame({'x1': np.random.rand(N),
                   'x2': np.random.rand(N),
                   'x3': np.random.rand(N),
                   'x4': np.random.rand(N),
                   'x5': np.random.rand(N)})

# Output metric y is a function of others
XY['performance'] = XY['x1'] * 3*XY['x2']**2


#%%

### Use the model

## Boosted trees (nonlinear boundary)
SE = ScenarioExplorer.ScenarioExplorer(XY,
                                       fail_threshold = 0.5,
                                       fail_criteria = '<=',
                                       method = 'boosted-trees')

# Choose variables of interest
plot_variables = ['x1', 'x2']

SE.plot_contour(plot_variables,
                save_figure = False)


## Logistic regression (linear)
SE = ScenarioExplorer.ScenarioExplorer(XY,
                                       fail_threshold = 0.5,
                                       fail_criteria = '<=',
                                       method = 'logistic')

# Choose variables of interest
plot_variables = ['x1', 'x2']

SE.plot_contour(plot_variables,
                save_figure = False)

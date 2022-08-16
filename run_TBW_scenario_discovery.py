# -*- coding: utf-8 -*-
"""
Trevor Amestoy
Cornell University
Summer 2022

Explores the 1,000 realizations of demand uncertainty used in TBW model.

Methods:
    


"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided


from logistic_regression_functions import fit_logistic, rank_significance, normalize_columns


from scenario_discovery_library import LinearRegression


#%%

# Specify which runs to consider
consider_runs = ['145']

# Constants
years = range(20)
months = range(240)


# Set paths
performance_data_path = 'C:/Users/tja73/Box/TampaBayWater/PerformanceAssessment/TBW_Performance_Assessment/Supply_Assessment/scenario_discovery/realization_performance_data'
input_data_path = 'C:/Users/tja73/Box/TampaBayWater/PerformanceAssessment/TBW_Performance_Assessment/Supply_Assessment/scenario_discovery/realization_input_data'

# Laptop
performance_data_path = 'C:/Users/tjame/Box Sync/TampaBayWater\PerformanceAssessment/TBW_Performance_Assessment/Supply_Assessment/scenario_discovery/realization_performance_data'
input_data_path = 'C:/Users/tjame/Box Sync/TampaBayWater/PerformanceAssessment/TBW_Performance_Assessment/Supply_Assessment/scenario_discovery/realization_input_data'


#%%

### Load data 

for run in consider_runs:

    ## Load performance data
    shortfall_count = np.loadtxt(f'{performance_data_path}/{run}_monthly_shortfall_counts_all.csv', delimiter = ',')[:,0:-4]

    ## Load input data
    # Annual metrics
    min_annual_reservoir = np.loadtxt(f'{input_data_path}/min_annual_res_elev_0{run}.csv', delimiter = ',')[:,0:-1]
    annual_demands = np.loadtxt(f'{input_data_path}/0{run}_annual_demands.csv', delimiter = ',')[:,0:-1]

    # Monthly metrics
    monthly_demands = np.loadtxt(f'{input_data_path}/0{run}_monthly_demands.csv', delimiter = ',')[:,0:-1]        
    monthly_avg_reservoir = np.loadtxt(f'{input_data_path}/0{run}_avg_monthly_reservoir_elev.csv', delimiter = ',')[:, 0:-1]
    monthly_min_reservoir = np.loadtxt(f'{input_data_path}/0{run}_min_monthly_reservoir_elev.csv', delimiter = ',')[:, 0:-1]





#%%

### Prepare data frame for SD




## Annual metrics
param_labs = ['Demands', 'Min. Reservoir']


sow_data = pd.DataFrame({'Demands' : monthly_demands.flatten(), 
                         'Min Reservoir' : monthly_min_reservoir.flatten(), 
                         'Avg. Reservoir' : monthly_avg_reservoir.flatten()})

# Test normalization
#ndf = normalize_columns(sow_data)
#sow_data['Intercept'] = 1

# Generate object
SD_model = LinearRegression(sow_data, shortfall_count.flatten(), param_labs, threshold = 7)


fit_model = SD_model.run()

#ranks = SD_model.rank_inputs()

SD_model.plot_parameter_contour_map('Avg. Reservoir', 'Min Reservoir')



"""
# TESTING
# Create a column of intercepts of value 1
sow_data['Intercept'] = np.ones(np.shape(sow_data)[0])


sow_data['Success'] = pass_cases


reg_model = fit_logistic(sow_data)

ranks = rank_significance(sow_data)
ranks"""



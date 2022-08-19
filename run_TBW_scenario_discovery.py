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


from scenario_discovery_library import LogisticRegression


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

### Load data (OROP)

for run in consider_runs:

    ## Load performance data
    shortfall_count = np.loadtxt(f'{performance_data_path}/{run}_monthly_shortfall_counts_all.csv', delimiter = ',')[:,0:-4]

    ## Load input data
    # Annual metrics
    # min_annual_reservoir = np.loadtxt(f'{input_data_path}/min_annual_res_elev_0{run}.csv', delimiter = ',')[:,0:-1]
    # annual_demands = np.loadtxt(f'{input_data_path}/0{run}_annual_demands.csv', delimiter = ',')[:,0:-1]

    # Monthly metrics
    monthly_demands = np.loadtxt(f'{input_data_path}/0{run}_monthly_demands.csv', delimiter = ',')[:,0:-1]
    monthly_avg_reservoir = np.loadtxt(f'{input_data_path}/0{run}_avg_monthly_reservoir_elev.csv', delimiter = ',')[:,0:-1]
    monthly_min_reservoir = np.loadtxt(f'{input_data_path}/0{run}_min_monthly_reservoir_elev.csv', delimiter = ',')[:,0:-1]

#%%

# Pre-SD exploration

values, base = np.histogram(shortfall_count.flatten(), bins = 40)

cumulative = np.cumsum(values)
norm_cumulative = (cumulative - min(cumulative)) / (max(cumulative) - min(cumulative)) * 100

a1 = plt.subplot()
a1.hist(shortfall_count.flatten(), bins = 40, color = 'orange')
a1.set_ylabel('Count across 1,000 realizations')

a2 = a1.twinx()
a2.plot(base[:-1], norm_cumulative)
a2.set_ylabel('Cumulative distribution (%)')
plt.xlabel('Shortfall duration (days)')

#%%

### Prepare data frame for SD

# Generate data frame of monthly inputs
sow_data = pd.DataFrame({'Demands' : monthly_demands.flatten(),
                         'Min. Reservoir' : monthly_min_reservoir.flatten(),
                         'Avg. Reservoir' : monthly_avg_reservoir.flatten()})

# Generate object
SD_model = LogisticRegression(sow_data, shortfall_count.flatten(), threshold = 7)

# Fit the model
parameters_of_interest = ['Demands', 'Min. Reservoir']
fit_model = SD_model.fit_logistic(subset_predictors=True, subset = parameters_of_interest)

# Rank the parameters by significance
# ranks = SD_model.rank_inputs()

# Plot the success contour map for a single parameter pair

SD_model.plot_parameter_contour_map(variable_params = parameters_of_interest)

# Check the change in SOS
SD_model.plot_area_SOS(variable_params = parameters_of_interest, threshold_range = [1,29])

#%%

### SD using SWRE inputs (streamflow and precip)

## Monthly data

gage_locations = ['ALA', 'MB', 'CYC', 'STL', 'PC']

# Load monthly input (SWRE)
ALA_in = np.loadtxt(f'./TBW_SD_data/ALA_monthly_data.csv', delimiter = ',')[:,0:-1]
MB_in = np.loadtxt(f'./TBW_SD_data/MB_monthly_data.csv', delimiter = ',')[:,0:-1]
CYC_in = np.loadtxt(f'./TBW_SD_data/CYC_monthly_data.csv', delimiter = ',')[:,0:-1]
STL_in = np.loadtxt(f'./TBW_SD_data/STL_monthly_data.csv', delimiter = ',')[:,0:-1]
PC_in = np.loadtxt(f'./TBW_SD_data/PC_monthly_data.csv', delimiter = ',')[:,0:-1]

# Arrange dataframe
data_in = pd.DataFrame({'Alafia Flow': ALA_in.flatten(),
                        'MB Flow': MB_in.flatten(),
                        'CYC Rain': CYC_in.flatten(),
                        'STL Rain': STL_in.flatten(),
                        'PC Rain': PC_in.flatten(),
                        'Demands': monthly_demands.flatten()})

# Make object
scenario_model = LogisticRegression(data_in, shortfall_count.flatten(), threshold = 15)

# Plot combinations of interest
poi = [['Demands','Alafia Flow'],
        ['Demands', 'MB Flow'],
        ['Alafia Flow', 'MB Flow'],
        ['Demands', 'CYC Rain']]

for vp in poi:
    scenario_model.plot_parameter_contour_map(variable_params = vp)

















"""
# TESTING
# Create a column of intercepts of value 1
sow_data['Intercept'] = np.ones(np.shape(sow_data)[0])


sow_data['Success'] = pass_cases


reg_model = fit_logistic(sow_data)

ranks = rank_significance(sow_data)
ranks"""

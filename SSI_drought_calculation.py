"""
Trevor Amestoy
August 2022
Cornell University

Used to identify periods of drought using SSI6.
"""

import numpy as np
import matplotlib.pyplot as plt


################################################################################


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


################################################################################


def find_droughts(data, historical = None, timestep = 'week'):
    """
    data : array
        The timeseries of weekly streamflow to consider.
    historical : array
        A timeseries of historical data, used to calculate the mean/sd of
        streamflow for normalization. Default = None. If none is provided, then
        the mean and standard deviation of 'data' will be used to normalize.

    Returns:
    --------
    drought_periods : array
        The weeks (numbers) where SSI6 drought was met.
    """

    # Verify dimensions
    if len(data.shape) != 1:
        raise Exception('Data must be 1D, but is not.')

    if timestep == 'week':
        n_time = 24
    elif timestep == 'month':
        n_time = 6
    elif timestep == 'day':
        n_time = 168


    # Calculate mean and SD
    if historical is not None:
        mean = np.mean(historical)
        sd = np.std(historical)

    else:
        mean = np.mean(data)
        sd = np.std(data)

    # Noramlize data
    norm_flow = (data - mean) / sd

    # Calculate SSI index (mov avg over 24 weeks)
    SSI = running_mean(norm_flow, n_time)

    # Constants
    n_weeks = len(data)

    # Initialize storage
    drought_periods = []
    drought_binary = np.zeros(n_weeks)

    # Set a counter to keep track of continuous dry weeks
    dry_counter = 0         # Couts when SSI < 0
    very_dry_counter = 0    # Counts when SSI < -1

    for wk in range(int(n_time/2),len(SSI)):

        if SSI[wk] < 0:
            dry_counter += 1

            if SSI[wk] < -1:
                very_dry_counter += 1

        # Reset if not <0
        else:
            dry_counter = 0
            very_dry_counter = 0

        # Add week to the drought period record if conditions are met
        if (dry_counter >= n_time) and (very_dry_counter > 0):
            drought_binary[wk] = 1
            drought_periods.append(wk)

    return drought_periods, drought_binary, SSI


################################################################################


def plot_SSI_droughts(data, historical_data = None):

    # Calculate the drought periods
    periods, binary, SSI = find_droughts(data, historical = historical_data)

    #Plotting
    fig, ax = plt.subplots()

    ax.vlines(periods, ymin = -2, ymax = 2, colors = 'red', alpha = 0.3)
    ax.plot(range(len(SSI)), SSI, label = 'SSI6')

    ax.set_ylabel('SSI6')
    ax.set_xlabel('Time')
    ax.set_ylim([-2, 2])
    plt.show()
    return

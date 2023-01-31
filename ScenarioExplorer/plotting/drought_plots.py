"""
Trevor Amestoy
August 2022
Cornell University

Used to identify periods of drought using SSI6.

UNDER DEVELOPMENT: DO NOT USE
"""

import numpy as np
import matplotlib.pyplot as plt

from ScenarioExplorer.utils.metrics import find_droughts


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

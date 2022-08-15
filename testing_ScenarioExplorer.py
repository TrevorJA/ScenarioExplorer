


import pandas as pd
import matplotlib.pyplot as plt
from scenario_discovery_library import ScenarioExplorer

inputs = pd.DataFrame(np.random.rand(1000, 3), columns=["x1", "x2", "x3"])
response = inputs["x1"]*inputs["x2"] + 0.2*inputs["x3"]




model = ScenarioExplorer(inputs, response)

model.method = 'PRIM'

# set threshold
model.threshold = 0.75
model.threshold_type = '>'


model.run()

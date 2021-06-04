from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import numpy as np

experiment_id = "3ZVrRhamToO8Q567wASDpQ"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()

experiment_name = "logs-all-modules-anneal-"

df.run.unique()
run_df = df[df.run in experiment_names]

modules = list()
for tag in run_df.tag.unique():
    if tag.split('/')[0] == "Train":
        continue
    modules.append(tag.split('/')[1])

#Plot
num_rows = 3
fig, axs = plt.subplots(num_rows, len(modules)//num_rows, sharex=True, sharey=True, squeeze=False, figsize=(6,6))
for i, module in enumerate(modules):
    x = i%num_rows
    y = int(np.floor(i/num_rows))
    module_df = run_df[run_df.tag == '/'.join(["Val",module])]
    axs[x,y].plot(module_df.step, module_df.value)
    axs[x,y].set_xlabel("step")
    axs[x,y].set_ylabel("reward")
    axs[x,y].set_title(module)
plt.tight_layout()
plt.show()
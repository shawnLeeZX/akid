import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set()
sns.set_context("paper", rc={"lines.linewidth": 2})

data = pd.read_csv("./training_loss", sep=',')
val_data = pd.read_csv("./val_loss", sep=',')
x = np.linspace(0, 10, 61)
x_val = np.linspace(0, 10, 11)

fig, ax1 = plt.subplots()

plots = []

color = 'tab:red'
fontsize = 15
ax1.set_xlabel('Epoch', fontsize=fontsize)
ax1.set_ylabel('Loss', color=color, fontsize=fontsize)

p = ax1.plot(x, data["Loss"], color=color, label="Training Loss")
plots.extend(p)

# Set markers on epoch where we plot eigenspectrum
markers = [0, 4, 10]
marker_shapes = ['^', 'o', 's']
colors = ["purple", "orange", "green"]
for i, m in enumerate(markers):
    ax1.scatter(m, data["Loss"][m * 6], c=colors[i], marker=marker_shapes[i], s=200)

p = ax1.plot(x_val, val_data["Loss"], linestyle='--', color=color, label="Val Loss")
plots.extend(p)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(labelsize=fontsize)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color, fontsize=fontsize)  # we already handled the x-label with ax1
p = ax2.plot(x, data["Acc"], color=color, label="Train Acc")
plots.extend(p)
p = ax2.plot(x_val, val_data["Acc"], linestyle='--', color=color, label="Val Acc")
plots.extend(p)
ax2.tick_params(axis='y', labelcolor=color)
ax2.tick_params(labelsize=fontsize)

plt.legend(plots, [i.get_label() for i in plots], loc="center right",
        fontsize=fontsize)

fig.tight_layout()

plt.savefig("loss.png")

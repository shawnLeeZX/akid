import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
sns.set_context("paper", rc={"lines.linewidth": 2})

# sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# # Create the data
# rs = np.random.RandomState(1979)
# x = rs.randn(500)
# g = np.tile(list("ABCDEFGHIJ"), 50)
# df = pd.DataFrame(dict(x=x, g=g))
# m = df.g.map(ord)
# df["x"] += m

K = 1024
import pickle as pk
with open("./eigenspectrum.pk", 'rb') as f:
    data = pk.load(f)
psi_s = []
xs = []
colors = ["purple", "orange", "green"]
epochs = []
fig, ax = plt.subplots(figsize=(8, 6))
fontsize = 15
ax.set_xlabel('Eigenvalue', fontsize=fontsize)
ax.set_ylabel('Probability Density', fontsize=fontsize)
ax.tick_params(labelsize=fontsize)
# linestyles = ['-.', '--', '-']
linestyles = ['-', '-', '-']
marker_shapes = ['^', 'o', 's']
marker_position = [i for i in range(1024) if i % 50 == 0]
for i, v in enumerate(data):
    if i == 0 or i == 4 or i == 10:
        c,d = v[0]
        psi = v[1]
        x = np.linspace(-d + c, d + c, K)
        epoch = np.ones(K) * i

        ax.plot(x, psi, linestyle=linestyles[i//4], marker=marker_shapes[i//4], markersize=7, markevery=marker_position, color=colors[i//4], label="Epoch {}".format(i))
        plt.xscale("symlog")
        plt.yscale("log")
        # psi_s.append(psi)
        # xs.append(x)
        # epochs.append(epoch)

# psi_s = np.concatenate(psi_s)
# xs = np.concatenate(xs)
# epochs = np.concatenate(epochs)

# df = {"epoch": epochs, "psi": psi_s, "x": xs}
# df = pd.DataFrame(df)

# # Initialize the FacetGrid object
# # pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
# g = sns.FacetGrid(df, row="epoch", hue="epoch", aspect=3, height=3)

# # Draw the densities in a few steps
# g.map(plt.plot, "x", "psi")
# # g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
# plt.xscale("symlog")
# g.map(plt.axhline, y=0, lw=5)
# plt.yscale("log")
# g.map(plt.axvline, x=0, lw=5)


# # Define and use a simple function to label the plot in axes coordinates
# def label(x, color, label):
    # ax = plt.gca()
    # ax.text(0, .2, label, fontweight="bold", color=color,
            # ha="left", va="center", transform=ax.transAxes)


# g.map(label, "x")

# # Set the subplots to overlap
# g.fig.subplots_adjust(hspace=-0.7)

# # Remove axes details that don't play well with overlap
# g.set_titles("")
# # g.set(yticks=[])
# # g.despine(bottom=True, left=True)


plt.legend(fontsize=fontsize)
plt.savefig("eigenspectrum_final.png")

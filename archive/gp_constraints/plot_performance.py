import matplotlib.pyplot as plt
import numpy as np
import optuna
import json
import pickle

import optuna_integration

N_TRIALS = 100
N_SEEDS = 10


LABEL_DICT = {"tpe-mv": "Multivariate TPE", "tpe-uv": "TPE", "botorch": "BoTorch", "gp": "GP"}
COLOR_DICT = {"tpe-mv": "black", "tpe-uv": "black", "botorch": "firebrick", "gp": "mediumblue"}
MARKER_DICT = {"tpe-mv": "^", "tpe-uv": "D", "botorch": "s", "gp": "o"}
# MARKER_SIZE_DICT = {"tpe-mv": "5.5", "tpe-uv": "4.6", "botorch": "5.04", "gp": "5.3"}
# MARKER_SIZE_DICT = {"tpe-mv": 9.3, "tpe-uv": 7.5, "botorch": 9, "gp": 9}
MARKER_SIZE_DICT = {"tpe-mv": 9.3, "tpe-uv": 7.5, "botorch": 7.95, "gp": 9}


with open("data.pkl", "rb") as f:
    data = pickle.load(f)

dx = np.arange(N_TRIALS) + 1
fig, ax = plt.subplots(figsize=(10, 5))
lines = []
labels = []


for sampler_name, _values in data.items():
    if len(_values) == 0:
        continue

    color = COLOR_DICT[sampler_name]
    marker = MARKER_DICT[sampler_name]
    markersize = MARKER_SIZE_DICT[sampler_name]
    labels.append(LABEL_DICT[sampler_name])
    values = np.minimum.accumulate(_values, axis=-1)
    q75, meds, q25 = np.percentile(values, [75, 50, 25], axis=0)
    (line,) = ax.plot(
        dx,
        meds,
        color=color,
        marker=marker,
        markerfacecolor="none",
        markersize=markersize,
        markevery=4,
    )
    lines.append(line)
    ax.fill_between(dx, q25, q75, color=color, alpha=0.2)

ax.set_xlim(1, N_TRIALS)
ax.set_ylim(-1, 6.5)
ax.set_xlabel("Number of Trials", fontsize=16)
ax.set_ylabel("Feasible Objective Value", fontsize=16)
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.grid(which="minor", color="gray", linestyle=":")
ax.grid(which="major", color="black")
fig.legend(
    handles=lines,
    loc="lower center",
    labels=labels,
    bbox_to_anchor=(0.5, -0.1),
    fontsize=16,
    fancybox=False,
    ncol=len(lines),
)
plt.savefig("performance_all.png", bbox_inches="tight")

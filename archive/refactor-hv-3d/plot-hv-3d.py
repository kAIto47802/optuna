from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def plot_results(
    data: dict[str, np.ndarray],
    labels: dict[str, str],
    colors: dict[str, str],
    markers: dict[str, str],
    ylabel: str,
) -> Figure:

    fig, ax = plt.subplots()
    lines = []
    for name, d in data.items():
        mean = np.mean(d, axis=0)
        std = np.std(d, axis=0)
        dx = np.arange(mean.shape[0]) + 1
        (line,) = ax.plot(
            dx,
            mean,
            colors[name],
            label=labels[name],
            marker=markers[name],
            markevery=32,
            markerfacecolor="none",
        )
        lines.append(line)
        ax.fill_between(
            dx,
            mean - std,
            mean + std,
            alpha=0.2,
            color=colors[name],
        )
    ax.legend(handles=lines, loc="upper left", fontsize=8, labels=list(labels.values()))
    ax.set_xlabel("Number of Trials")
    ax.set_ylabel(ylabel)
    ax.grid(which="major", color="black")

    return fig


def main(args: Namespace) -> None:
    commits = {
        "29f19f1acb9f8ac55799ba7278a4ec105222f3ff": "Original Master (29f19f1acb9f8ac55799ba7278a4ec105222f3ff)",
        "1ab5c2e3fdc99b7eb3f4cae5c092cbf20b8985b7": "Latest Master (1ab5c2e3fdc99b7eb3f4cae5c092cbf20b8985b7)",
        "0b43c3384d37d79dcc45a810d57fcdf8e4bee811": "This PR (0b43c3384d37d79dcc45a810d57fcdf8e4bee811)",
    }
    colors = dict(zip(commits.keys(), ["#0072B2", "#CC79A7", "#F0E442"]))
    markers = dict(zip(commits.keys(), ["o", "s", "D"]))

    times = {
        commit: np.load(
            f"{commit}_trial{args.n_trials}.npz",
        )["times"]
        for commit in commits.keys()
    }

    fig = plot_results(
        data=times,
        labels=commits,
        colors=colors,
        markers=markers,
        ylabel="Elapsed Time / s",
    )
    fig.savefig(
        f"time_trial{args.n_trials}.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--n_trials",
        type=int,
        default=1000,
        help="Number of trials for the optimization.",
    )
    args = parser.parse_args()
    main(args)


# 29f19f1acb9f8ac55799ba7278a4ec105222f3ff_trial1000: original
# 1ab5c2e3fdc99b7eb3f4cae5c092cbf20b8985b7_trial1000: latest
# 0b43c3384d37d79dcc45a810d57fcdf8e4bee811_trial1000: this PR

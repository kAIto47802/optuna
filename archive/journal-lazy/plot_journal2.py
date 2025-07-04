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
            markevery=1,
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
        "29f19f1acb9f8ac55799ba7278a4ec105222f3ff": "Original (29f19f1acb9f8ac55799ba7278a4ec105222f3ff)",
        "eace0a44b40b4487b69d373b18613d0fe942536a": "This PR (eace0a44b40b4487b69d373b18613d0fe942536a)",
    }
    colors = dict(zip(commits.keys(), ["#0072B2", "#CC79A7"]))
    markers = dict(zip(commits.keys(), ["o", "s"]))

    times = {
        commit: np.load(
            f"{commit}_trial{args.n_trials}_dim{args.dimension}_n_jobs{args.n_jobs}_backend{args.backend}.npz"
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
        "--backend",
        choices=["file", "redis"],
        default="file",
        help="Storage backend for the Journal storage. Options are 'file' or 'redis'.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=5,
        help="Dimension (2, 3, 5, 10, 20, 40, 60) for the BBOB benchmark.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=1000,
        help="Number of trials for the optimization.",
    )
    args = parser.parse_args()
    main(args)

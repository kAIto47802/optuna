from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def _prepare_data(data: dict[str, np.ndarray | int]) -> tuple[np.ndarray, np.ndarray]:
    times = data["times"]
    values = data["values"]
    assert isinstance(times, np.ndarray) and isinstance(values, np.ndarray)
    assert times.shape == values.shape
    return times, values


def plot_results(
    data: dict[str, np.ndarray],
    colors: dict[str, str],
    markers: dict[str, str],
    ylabel: str,
    accumulate: bool = False,
) -> Figure:

    fig, ax = plt.subplots()
    lines = []
    for name, d in data.items():
        if accumulate:
            d = np.minimum.accumulate(d, axis=-1)
        mean = np.mean(d, axis=0)
        std = np.std(d, axis=0)
        dx = np.arange(mean.shape[0]) + 1
        (line,) = ax.plot(
            dx,
            mean,
            colors[name],
            label=name,
            marker=markers[name],
            markevery=4,
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
    ax.legend(
        handles=lines,
        loc="upper left",
        fontsize=8,
        labels=list(data.keys()),
    )
    ax.set_xlabel("Number of Trials")
    ax.set_ylabel(ylabel)
    ax.grid(which="major", color="black")

    return fig


def main(args: Namespace) -> None:
    colors = dict(
        zip(
            args.versions,
            [
                "#0072B2",
                "#CC79A7",
                "#F0E442",
                "#D55E00",
                "#009E73",
                "#E69F00",
                "#56B4E9",
                "#000000",
            ][: len(args.versions)],
        )
    )
    markers = dict(
        zip(args.versions, ["o", "s", "D", "^", "v", "P", "*", "X"][: len(args.versions)])
    )

    data = [
        np.load(
            f"results/bbob_fn{args.function_id}_dim{args.dimension}_scipy{v.replace('v', '')}_trial{args.n_trials}.npz",
        )
        for v in args.versions
    ]

    times, values = zip(*[_prepare_data(d) for d in data])

    fig = plot_results(
        dict(zip(args.versions, times)),
        colors=colors,
        markers=markers,
        ylabel="Elapsed Time / s",
        accumulate=False,
    )
    fig.savefig(
        f"results/times_bbob_fn{args.function_id}_dim{args.dimension}_{'-'.join(args.versions)}.png",
        bbox_inches="tight",
    )

    fig = plot_results(
        dict(zip(args.versions, values)),
        colors=colors,
        markers=markers,
        ylabel="Function Value",
        accumulate=True,
    )
    fig.savefig(
        f"results/values_bbob_fn{args.function_id}_dim{args.dimension}_{'-'.join(args.versions)}.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--function_id",
        type=int,
        default=3,
        help="Function ID (1-24) for the BBOB benchmark.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=5,
        help="Dimension (2, 3, 5, 10, 20, 40, 60) for the BBOB benchmark.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials for the optimization.",
    )
    parser.add_argument(
        "--versions",
        type=str,
        nargs="+",
        default=["v1.14.0", "v1.15.0"],
        help="List of scipy versions to compare.",
    )
    args = parser.parse_args()
    main(args)

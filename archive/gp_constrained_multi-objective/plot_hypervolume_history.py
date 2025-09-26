from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure

import optuna
from optuna.visualization._hypervolume_history import (
    _get_hypervolume_history_info,
    _HypervolumeHistoryInfo,
)

fp = font_manager.FontProperties(fname="/usr/share/fonts/TTF/Times.TTF")


def plot_results(
    data: dict[str, list[_HypervolumeHistoryInfo]],
    names: dict[str, str],
    colors: dict[str, str],
    markers: dict[str, str],
    marker_sizes: dict[str, float],
    legend_order: list[str],
    ylabel: str,
    use_sf: bool = True,
    trial_min: int | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    lines = {}
    names_list = {}
    for name, d in data.items():
        values = np.array([info.values for info in d])
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0) / np.sqrt(values.shape[0])
        dx = d[0].trial_numbers
        (line,) = ax.plot(
            dx[trial_min:],
            mean[trial_min:],
            colors[name],
            label=names[name],
            marker=markers[name],
            markersize=marker_sizes[name] * 1.2,
            markevery=8,
        )
        lines[name] = line
        names_list[name] = names[name]
        ax.fill_between(
            dx[trial_min:],
            (mean - std)[trial_min:],
            (mean + std)[trial_min:],
            alpha=0.2,
            color=colors[name],
        )
    ax.legend(
        handles=[lines[name] for name in legend_order],
        labels=[names_list[name] for name in legend_order],
        loc="lower right",
        fontsize=12,
        prop=(
            font_manager.FontProperties(fname="/usr/share/fonts/TTF/Times.TTF", size=12)
            if use_sf
            else None
        ),
    )
    ax.set_xlabel("Number of Trials", fontsize=13, fontproperties=fp if use_sf else None)
    ax.set_ylabel(ylabel, fontsize=13, fontproperties=fp if use_sf else None)

    ax.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
    if use_sf:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(fp)
    ax.tick_params(labelsize=12)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig


def main(args: Namespace) -> None:
    sampler_names = {
        "gp": "GPSampler",
        "tpe": "TPESampler",
        "nsgaii": "NSGAIISampler",
    }
    colors = {
        "gp": "#0072B2",
        "tpe": "#CC79A7",
        "nsgaii": "#E69F00",
    }
    markers = {
        "gp": "*",
        "tpe": "o",
        "nsgaii": "D",
    }
    marker_sizes = {
        "gp": 8.0,
        "tpe": 5.1,
        "nsgaii": 4.2,
    }
    legend_order = ["gp", "tpe", "nsgaii"]

    name = f"cdtlz_C{args.constraint_type}-DTLZ{args.function_id}_n_objectives{args.n_objectives}_dim{args.dimension}_trial{args.n_trials}"
    data = {
        sampler_name: [
            _get_hypervolume_history_info(
                optuna.load_study(
                    study_name=f"{name}_{sampler_name}_seed{seed}",
                    storage="sqlite:///results/results.db",
                ),
                np.array([2.0] * args.n_objectives, dtype=float),
            )
            for seed in range(42, 42 + args.n_seeds)
        ]
        for sampler_name in args.samplers
    }

    fig = plot_results(
        data,
        names=sampler_names,
        colors=colors,
        markers=markers,
        marker_sizes=marker_sizes,
        legend_order=legend_order,
        ylabel="Hypervolume",
        use_sf=args.use_sf,
        xlim=(6, args.n_trials + 4),
        ylim=(2.56, 3.23),
        trial_min=10,
        figsize=(7.5, 4),
    )
    fig.savefig(
        f"results/{name}_hypervolume_history{'_sf' if args.use_sf else ''}.png",
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--samplers",
        type=str,
        nargs="+",
        default=["nsgaii", "tpe", "gp"],
        help="Sampler to use for the optimization.",
    )
    parser.add_argument(
        "--n_objectives",
        type=int,
        default=2,
        help="Number of objectives for the problem.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=3,
        help="Dimension of the problem (number of variables).",
    )
    parser.add_argument(
        "--constraint_type",
        type=int,
        default=1,
        help="Constraint type for the problem (1 or 2).",
    )
    parser.add_argument(
        "--function_id",
        type=int,
        default=2,
        help="Function ID for the DTLZ problem (1-7).",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=300,
        help="Number of trials for the optimization.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of random seeds for the optimization.",
    )
    parser.add_argument(
        "--use_sf",
        type=bool,
        default=True,
        help="Use sans-serif font for the plot.",
    )
    args = parser.parse_args()

    main(args)

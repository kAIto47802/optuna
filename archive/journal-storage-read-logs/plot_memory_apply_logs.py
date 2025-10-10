from argparse import ArgumentParser, Namespace
import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import polars as pl

fp = font_manager.FontProperties(fname="/usr/share/fonts/TTF/Times.TTF")


def _prepare_data(branch: str) -> pl.DataFrame:
    df = pl.read_csv(f"results/apply_logs_{branch}.csv")
    return df.group_by("length").agg(
        [
            pl.col("before").mean().alias("before_mean"),
            (pl.col("before").std() / pl.count("before").cast(pl.Float64).sqrt()).alias("before_se"),
            pl.col("after").mean().alias("after_mean"),
            (pl.col("after").std() / pl.count("after").cast(pl.Float64).sqrt()).alias("after_se"),
        ]
    ).sort("length")

def plot_results(
    data: dict[str, np.ndarray],
    colors: dict[str, str],
    markers: dict[str, str],
    marker_sizes: dict[str, float],
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for name, d in data.items():
        x, mean, se = d.T
        ax.plot(
            x,
            mean,
            colors[name],
            label=name,
            marker=markers[name],
            markersize=marker_sizes[name] * 1.2,
        )
        ax.fill_between(
            x,
            (mean - se),
            (mean + se),
            alpha=0.2,
            color=colors[name],
        )
    ax.legend(
        loc="upper left",
        fontsize=12,
        prop=(
            font_manager.FontProperties(fname="/usr/share/fonts/TTF/Times.TTF", size=12)
        ),
    )
    ax.set_xlabel(xlabel, fontsize=13, fontproperties=fp)
    ax.set_ylabel(ylabel, fontsize=13, fontproperties=fp)

    ax.set_xscale("log")

    ax.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(fp)
    ax.tick_params(labelsize=12)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig


def main(args: Namespace) -> None:
    dfs = [_prepare_data(branch) for branch in args.branches]
    names = ["Original", "This PR"]
    colors = {
        "Original": "#CC79A7",
        "This PR": "#0072B2",
    }
    maekers = {
        "Original": "o",
        "This PR": "*",
    }
    marker_sizes = {
        "Original": 6.0,
        "This PR": 8.0,
    }
    for phase in ["before", "after"]:
        data = {
            name:
            df.select(
                [
                    pl.col("length").cast(pl.Float64),
                    pl.col(f"{phase}_mean").cast(pl.Float64),
                    pl.col(f"{phase}_se").cast(pl.Float64),
                ]
            ).to_numpy()
            for name, df in zip(names, dfs)
        }
        fig = plot_results(
            data=data,
            colors=colors,
            markers=maekers,
            marker_sizes=marker_sizes,
            xlabel="Log file length",
            ylabel="Memory usage / MB",
            figsize=(6, 4),
        )
        fig.savefig(f"results/memory_apply_logs_{phase}.png", dpi=300, bbox_inches="tight")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--branches", type=str, nargs="+")
    args = parser.parse_args()

    main(args)
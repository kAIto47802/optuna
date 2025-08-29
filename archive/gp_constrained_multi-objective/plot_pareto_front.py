from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import optunahub
from matplotlib import font_manager
from matplotlib.figure import Figure

import optuna
import optuna.visualization.matplotlib
from optuna.visualization._pareto_front import _ParetoFrontInfo
from optuna.visualization.matplotlib import plot_pareto_front

fp = font_manager.FontProperties(fname="/usr/share/fonts/TTF/Times.TTF")


def _get_pareto_front_2d_patched(info: _ParetoFrontInfo, use_sf: bool = True) -> Figure:
    # Set up the graph style.
    # plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig, ax = plt.subplots()

    ax.set_xlabel(
        info.target_names[info.axis_order[0]], fontsize=13, fontproperties=fp if use_sf else None
    )
    ax.set_ylabel(
        info.target_names[info.axis_order[1]], fontsize=13, fontproperties=fp if use_sf else None
    )

    if len(info.infeasible_trials_with_values) > 0:
        ax.scatter(
            x=[values[info.axis_order[0]] for _, values in info.infeasible_trials_with_values],
            y=[values[info.axis_order[1]] for _, values in info.infeasible_trials_with_values],
            color="#cccccc",
            alpha=0.6,
            label="Infeasible Trial",
        )
    if len(info.non_best_trials_with_values) > 0:
        ax.scatter(
            x=[values[info.axis_order[0]] for _, values in info.non_best_trials_with_values],
            y=[values[info.axis_order[1]] for _, values in info.non_best_trials_with_values],
            color="#0072B2",
            alpha=0.6,
            label="Feasible Trial",
        )
    if len(info.best_trials_with_values) > 0:
        ax.scatter(
            x=[values[info.axis_order[0]] for _, values in info.best_trials_with_values],
            y=[values[info.axis_order[1]] for _, values in info.best_trials_with_values],
            color="#CC79A7",
            alpha=0.6,
            label="Best Trial",
        )

    if info.non_best_trials_with_values is not None and ax.has_data():
        ax.legend(
            handlelength=0.8,
            handletextpad=0.4,
            prop=(
                font_manager.FontProperties(fname="/usr/share/fonts/TTF/Times.TTF", size=11.5)
                if use_sf
                else None
            ),
        )

    ax.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
    if use_sf:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(fp)
    ax.tick_params(labelsize=12)

    return fig


def main(args: Namespace) -> None:
    optuna.visualization.matplotlib._pareto_front._get_pareto_front_2d = (
        lambda info: _get_pareto_front_2d_patched(info, use_sf=args.use_sf)
    )

    cdtlz = optunahub.load_local_module(
        "dtlz_constrained", registry_root="../optunahub-registry/package/benchmarks"
    )
    problem = cdtlz.Problem(
        n_objectives=args.n_objectives,
        dimension=args.dimension,
        function_id=args.function_id,
        constraint_type=args.constraint_type,
    )

    name = f"cdtlz_C{args.constraint_type}-DTLZ{args.function_id}_n_objectives{args.n_objectives}_dim{args.dimension}_trial{args.n_trials}_{args.sampler}_seed{args.seed}"
    study = optuna.load_study(
        study_name=name,
        storage="sqlite:///results/results.db",
    )
    fig = plot_pareto_front(study, constraints_func=problem.constraints_func)
    # fig.savefig(f"results/{name}_pareto_front.png", bbox_inches="tight")
    fig.savefig(
        f"results/{name}_pareto_front{'_sf' if args.use_sf else ''}.png",
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--sampler",
        type=str,
        default="gp",
        help="Sampler to use for the optimization (default: GP).",
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
        "--seed",
        type=int,
        default=42,
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

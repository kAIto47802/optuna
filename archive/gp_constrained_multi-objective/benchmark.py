from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import optunahub

import optuna


def _extract_elapsed_time(study: optuna.study.Study) -> list[float]:
    return [
        (
            cast(datetime, t.datetime_complete) - cast(datetime, study.trials[0].datetime_start)
        ).total_seconds()
        for t in study.trials
    ]


def _extract_objective_value(study: optuna.study.Study) -> list[float | None]:
    return [t.values for t in study.trials]


def experiment_once(
    problem: optunahub.benchmarks.BaseProblem,
    sampler_name: str,
    n_trials: int,
    seed: int,
    name: str,
) -> tuple[list[float], list[float | None]]:
    sampler = {
        "gp": optuna.samplers.GPSampler(
            seed=seed, constraints_func=problem.constraints_func, deterministic_objective=True
        ),
        "gp_wo_constraints": optuna.samplers.GPSampler(seed=seed, deterministic_objective=True),
        "tpe": optuna.samplers.TPESampler(seed=seed, constraints_func=problem.constraints_func),
        "tpe_wo_constraints": optuna.samplers.TPESampler(seed=seed),
        "nsgaii": optuna.samplers.NSGAIISampler(
            seed=seed, constraints_func=problem.constraints_func
        ),
    }[sampler_name]
    study = optuna.create_study(
        study_name=f"{name}_seed{seed}",
        sampler=sampler,
        directions=problem.directions,
        storage="sqlite:///results/results.db",
    )
    study.optimize(problem, n_trials=n_trials)
    times = _extract_elapsed_time(study)
    values = _extract_objective_value(study)
    return times, values


def main(args: Namespace) -> None:
    Path("results").mkdir(exist_ok=True)
    cdtlz = optunahub.load_local_module(
        "dtlz_constrained", registry_root="../optunahub-registry/package/benchmarks"
    )
    problem = cdtlz.Problem(
        n_objectives=args.n_objectives,
        dimension=args.dimension,
        function_id=args.function_id,
        constraint_type=args.constraint_type,
    )

    name = f"cdtlz_C{args.constraint_type}-DTLZ{args.function_id}_n_objectives{args.n_objectives}_dim{args.dimension}_trial{args.n_trials}_{args.sampler}"
    times, values = zip(
        *[
            experiment_once(problem, args.sampler, args.n_trials, seed, name)
            for seed in range(42, 42 + args.n_seeds)
        ]
    )
    np.savez(
        f"results/{name}.npz",
        times=np.array(times),
        values=np.array(values),
        dimension=args.dimension,
        n_objectives=args.n_objectives,
        constraint_type=args.constraint_type,
        function_id=args.function_id,
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
        "--n_seeds",
        type=int,
        default=5,
        help="Number of random seeds for the optimization.",
    )
    args = parser.parse_args()

    main(args)

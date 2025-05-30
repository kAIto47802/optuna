from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import cast
import numpy as np
import scipy
import optuna
import optunahub


def _extract_elapsed_time(study: optuna.study.Study) -> list[float]:
    return [
        (
            cast(datetime, t.datetime_complete) - cast(datetime, study.trials[0].datetime_start)
        ).total_seconds()
        for t in study.trials
    ]


def _extract_objective_value(study: optuna.study.Study) -> list[float | None]:
    return [t.value for t in study.trials]


def experiment_once(
    objective: optunahub.benchmarks.BaseProblem,
    n_trials: int,
    seed: int,
) -> tuple[list[float], list[float | None]]:
    sampler = optuna.samplers.GPSampler(seed=seed)
    study = optuna.create_study(sampler=sampler, directions=objective.directions)
    study.optimize(objective, n_trials=n_trials)
    times = _extract_elapsed_time(study)
    values = _extract_objective_value(study)
    return times, values


def main(args: Namespace) -> None:
    bbob = optunahub.load_module("benchmarks/bbob")

    objective = bbob.Problem(function_id=args.function_id, dimension=args.dimension)

    times, values = zip(
        *[experiment_once(objective, args.n_trials, seed) for seed in range(args.n_seeds)]
    )
    np.savez(
        f"results/bbob_fn{args.function_id}_dim{args.dimension}_scipy{scipy.__version__}_trial{args.n_trials}.npz",
        times=np.array(times),
        values=np.array(values),
        function_id=args.function_id,
        dimension=args.dimension,
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
        "--n_seeds",
        type=int,
        default=5,
        help="Number of random seeds for the optimization.",
    )
    args = parser.parse_args()

    main(args)

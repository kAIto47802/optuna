from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import cast

import numpy as np
import optuna


def objective(trial: optuna.Trial) -> float:
    _x = trial.suggest_float("x", 0.0, 10.0, step=0.1)
    _y = trial.suggest_int("y", 0, 10)
    return 0.0


def _extract_elapsed_time(study: optuna.study.Study) -> list[float]:
    return [
        (
            cast(datetime, t.datetime_complete) - cast(datetime, study.trials[0].datetime_start)
        ).total_seconds()
        for t in study.trials
    ]


def _measure_time(
    sampler: optuna.samplers.BaseSampler,
    n_trials: int = 100,
) -> list[float]:
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return _extract_elapsed_time(study)


def main(args: Namespace) -> None:
    samplers = {
        "tpe": optuna.samplers.TPESampler,
        "brute-force": optuna.samplers.BruteForceSampler,
    }

    data = {
        k: np.array(
            [
                _measure_time(sampler=sampler_cls(seed=42 + s), n_trials=args.n_trials)
                for s in range(args.n_seeds)
            ]
        )
        for k, sampler_cls in samplers.items()
    }

    mean = {k: np.mean(v, axis=0) for k, v in data.items()}
    std = {k: np.std(v, axis=0) for k, v in data.items()}

    np.savez(f"mean{args.suffix}.npz", **mean)
    np.savez(f"std{args.suffix}.npz", **std)
    np.savez(
        f"experimental_settings{args.suffix}.npz",
        n_trials=args.n_trials,
        n_seeds=args.n_seeds,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials to run for each sampler.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of seeds to use for each sampler.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to append to the output file names.",
    )
    args = parser.parse_args()

    main(args)

from argparse import ArgumentParser, Namespace
from datetime import datetime
import subprocess
from typing import cast

import numpy as np
import optuna


def _extract_elapsed_time(study: optuna.study.Study) -> list[float]:
    return [
        (
            cast(datetime, t.datetime_complete) - cast(datetime, study.trials[0].datetime_start)
        ).total_seconds()
        for t in study.trials
    ]


def _get_git_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")


def objective(trial: optuna.Trial) -> tuple[float, float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2, (x + 2) ** 2 + (y + 2) ** 2


def experiment_once(
    n_trials: int,
    seed: int,
) -> list[float]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, directions=["minimize"] * 3)
    study.optimize(objective, n_trials=n_trials)
    times = _extract_elapsed_time(study)
    return times


def main(args: Namespace) -> None:
    name = f"{_get_git_commit_hash()}_trial{args.n_trials}"
    times = [experiment_once(args.n_trials, seed) for seed in range(42, 42 + args.n_seeds)]
    np.savez(
        f"{name}.npz",
        times=np.array(times),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--n_trials",
        type=int,
        default=1000,
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

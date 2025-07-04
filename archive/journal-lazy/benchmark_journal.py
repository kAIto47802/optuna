from argparse import ArgumentParser, Namespace
from datetime import datetime
import subprocess
from typing import cast, Literal

import numpy as np
import optuna
import optunahub


def _extract_elapsed_time(study: optuna.study.Study) -> list[float]:
    return [
        (
            cast(datetime, t.datetime_complete) - cast(datetime, study.trials[0].datetime_start)
        ).total_seconds()
        for t in study.trials
    ]


def _get_git_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")


def experiment_once(
    objective: optunahub.benchmarks.BaseProblem,
    backend: Literal["file", "redis"],
    url: str,
    n_trials: int,
    seed: int,
) -> list[float]:
    sampler = optuna.samplers.RandomSampler(seed=seed)
    storage = optuna.storages.JournalStorage(
        {
            "file": optuna.storages.journal.JournalFileBackend,
            "redis": optuna.storages.journal.JournalRedisBackend,
        }[backend](url)
    )
    study = optuna.create_study(sampler=sampler, storage=storage, directions=objective.directions)
    study.optimize(objective, n_trials=n_trials)
    times = _extract_elapsed_time(study)
    return times


def main(args: Namespace) -> None:
    bbob = optunahub.load_module("benchmarks/bbob")
    objective = bbob.Problem(function_id=args.function_id, dimension=args.dimension)

    name = f"{_get_git_commit_hash()}_trial{args.n_trials}_fn{args.function_id}_dim{args.dimension}_n_jobs{args.n_jobs}"
    times = [
        experiment_once(
            objective,
            args.backend,
            args.url or f"{name}_seed{seed}.log",
            args.n_trials,
            seed,
        )
        for seed in range(42, 42 + args.n_seeds)
    ]
    np.savez(
        f"{name}_backend{args.backend}.npz",
        times=np.array(times),
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
        "--url",
        type=str,
        default=None,
        help="URL for the Journal storage backend",
    )
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
        default=1000,
        help="Number of trials for the optimization.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of random seeds for the optimization.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run.",
    )
    args = parser.parse_args()

    main(args)

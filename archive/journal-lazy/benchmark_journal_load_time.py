from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from datetime import datetime
import time
import subprocess
from typing import cast, Literal

import numpy as np
import optuna
import optunahub


def _get_git_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")


def experiment_once(
    objective: Callable[[optuna.Trial], float],
    backend: Literal["file", "redis"],
    url: str,
    seed: int,
) -> tuple[float, float, float]:
    sampler = optuna.samplers.RandomSampler(seed=seed)
    start0 = time.time()
    storage = optuna.storages.JournalStorage(
        {
            "file": optuna.storages.journal.JournalFileBackend,
            "redis": optuna.storages.journal.JournalRedisBackend,
        }[backend](url)
    )
    start = time.time()
    study = optuna.create_study(sampler=sampler, storage=storage)
    mid = time.time()
    study.optimize(objective, n_trials=1)
    end = time.time()
    return start - start0, mid - start, end - mid


def main(args: Namespace) -> None:

    def objective(trial: optuna.Trial) -> float:
        x = [trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(args.dimension)]
        return sum(a**2 for a in x)

    name = f"{_get_git_commit_hash()}_trial1_dim{args.dimension}_n_jobs{args.n_jobs}"
    times = [
        experiment_once(
            objective,
            args.backend,
            args.url or f"{name}_seed{seed}.log",
            seed,
        )
        for seed in range(42, 42 + args.n_seeds)
    ]
    print(times)
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
        "--dimension",
        type=int,
        default=5,
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


# n_jobs=1

# This PR
# [(0.01573967933654785, 0.050244808197021484), (0.015886545181274414, 0.049056053161621094), (0.016665220260620117, 0.04921412467956543), (0.016794681549072266, 0.050322771072387695), (0.016590595245361328, 0.047841787338256836)]

# Original
# [(0.01585984230041504, 0.04702305793762207), (0.01539921760559082, 0.04829764366149902), (0.015573263168334961, 0.049219608306884766), (0.015501737594604492, 0.048715829849243164), (0.015562057495117188, 0.04773736000061035)]


# n_jobs=50

# This PR
# [(11.937805652618408, 0.015875577926635742, 0.048163414001464844), (11.68703842163086, 0.015820741653442383, 0.048681020736694336), (11.811894178390503, 0.01631760597229004, 0.05131125450134277), (11.83766484260559, 0.015448808670043945, 0.050112247467041016), (12.09284496307373, 0.015701770782470703, 0.04931926727294922)]

# Original
# [(11.835241794586182, 0.015510082244873047, 0.04874825477600098), (11.818103075027466, 0.015735626220703125, 0.04861593246459961), (11.89995789527893, 0.015478849411010742, 0.04772615432739258), (11.824550867080688, 0.01578831672668457, 0.04881477355957031), (11.918012857437134, 0.015418291091918945, 0.04832959175109863)]

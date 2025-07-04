from argparse import ArgumentParser, Namespace
from datetime import datetime

import optuna


def main(args: Namespace) -> None:
    def objective(trial: optuna.Trial) -> float:
        x = [trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(args.dimension)]
        return sum(a**2 for a in x)

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")

    sampler = optuna.samplers.RandomSampler(seed=args.seed)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            f"./dim{args.dimension}_trials{args.n_trials}_seed{args.seed}_{timestamp}.log"
        )
    )
    study = optuna.create_study(sampler=sampler, storage=storage)
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dimension",
        type=int,
        default=10,
        help="Number of dimensions for the optimization problem.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100000,
        help="Number of trials to run in the optimization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=31,
        help="Number of parallel jobs to run.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Timestamp for the output file name. If not provided, current time will be used.",
    )
    args = parser.parse_args()
    main(args)

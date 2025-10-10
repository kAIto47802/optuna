import optuna
import argparse


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

def main(args: argparse.Namespace) -> None:
    optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f"./journal_storage{args.log_length or ''}.log")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_length", type=int)
    args = parser.parse_args()
    main(args)

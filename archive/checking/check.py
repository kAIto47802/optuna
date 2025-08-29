from collections.abc import Callable, Sequence

import numpy as np

import optuna
from optuna.study._study_direction import StudyDirection
from optuna.study.study import ObjectiveFuncType


def single_objective(trial: optuna.Trial) -> float:
    return sum(multi_objective(trial))


def multi_objective(trial: optuna.Trial) -> tuple[float, float]:
    a = trial.suggest_float("a", -10.0, 20.0)
    b = trial.suggest_float("b", 1.0, 100.0, log=True)
    c = trial.suggest_float("c", -10.0, 20.0, step=0.1)
    d = trial.suggest_int("d", 3, 10)
    e = trial.suggest_int("e", 1, 100, log=True)
    f = trial.suggest_categorical("f", ["x", "y", "z"])
    trial.set_user_attr("a+b", a + b)
    trial.set_user_attr("c+d", c + d)
    return a + b + c, d + e + ord(f)


def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float, float]:
    ab = trial.user_attrs["a+b"]
    cd = trial.user_attrs["c+d"]
    return ab - 10.0, cd - 5.0


def experiment(
    objective: ObjectiveFuncType,
    directions: Sequence[str | StudyDirection] | None = None,
    constraints_func: Callable[[optuna.trial.FrozenTrial], Sequence[float]] | None = None,
) -> None:
    sampler = optuna.samplers.GPSampler(
        n_startup_trials=1, seed=42, constraints_func=constraints_func
    )
    study = optuna.create_study(sampler=sampler, directions=directions)
    study.optimize(objective, n_trials=30)

    for trial in study.trials:
        print(f"Trial {trial.number}: {trial.values} (params: {trial.params})")


def main() -> None:
    print("Single Objective Optimization:")
    experiment(single_objective)
    print("Constrained Optimization:")
    experiment(single_objective, constraints_func=constraints)
    print("Multi-objective Optimization:")
    experiment(multi_objective, directions=["minimize"] * 2)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import optuna


N_TRIALS = 100


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 0.0, 2 * np.pi)
    y = trial.suggest_float("y", 0.0, 2 * np.pi)
    return float(np.sin(x) + y)


def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    x = trial.params["x"]
    y = trial.params["y"]
    c = float(np.sin(x) * np.sin(y) + 0.95)
    trial.set_user_attr("c", c)
    return (c, )

# def objective(trial: optuna.Trial) -> float:
#     x = trial.suggest_float("x", 0.0, 2 * np.pi)
#     y = trial.suggest_float("y", 0.0, 2 * np.pi)
#     return float(np.cos(2 * x) * np.cos(y) + np.sin(x))

# def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
#     x = trial.params["x"]
#     y = trial.params["y"]
#     c = float(np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y) - 0.5)
#     trial.set_user_attr("c", c)
#     return (c,)




sampler = optuna.samplers.GPSampler(seed=42, constraints_func=constraints)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)





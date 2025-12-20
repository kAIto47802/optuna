import optuna
from optuna.visualization import plot_param_importances
from optuna.importance._ped_anova.evaluator import PedAnovaImportanceEvaluator as ConditionalPedAnovaImportanceEvaluator
from optuna.importance._ped_anova.orig_evaluator import PedAnovaImportanceEvaluator
import random

# def objective(trial: optuna.trial.Trial) -> float:
#     x = trial.suggest_float('x', -5.0, 5.0)
#     # w = trial.suggest_float('w', -1.0, 1.0)
#     w = 1
#     # c = trial.suggest_categorical('c', [True, False])
#     c = trial.suggest_float("c", 0, 1)
#     if c> 0.5:
#         y = trial.suggest_float('y', -10.0, 10.0)
#         # z = trial.suggest_float('z', -1.0, 1.0)
#         z = 0
#         return x - (y < -9) * 100
#     return x


# def objective(trial: optuna.trial.Trial) -> float:
#     c = trial.suggest_float("c", 0.0, 1.0)
#     noize = + random.uniform(-5, 5)
#     if c < 0.5:
#         x = trial.suggest_float('x', -10.0, 0.0)
#         return x + noize
#     else:
#         y = trial.suggest_float('y', 0.0, 10.0)
#         return y + noize


def objective(trial: optuna.trial.Trial) -> float:
    c = trial.suggest_float("c", 0.0, 1.0)
    noize = random.uniform(-10, 10)
    # noize = 0
    if c < 0.5:
        x = trial.suggest_float('x', -20.0, 10.0)
        return x + noize
    else:
        y = trial.suggest_float('y', -10.0, 20.0)
        return y + noize


# def objective(trial: optuna.trial.Trial) -> float:
#     c = trial.suggest_float("c", 0.0, 1.0)
#     noize = random.uniform(-10, 10)
#     if c < 0.5:
#         x = trial.suggest_float('x', -10.0, 10.0)
#         return x + noize
#     else:
#         y = trial.suggest_float('y', -10.0, 10.0)
#         return y + noize



def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float('x', 0.0, 10.0)
    y = trial.suggest_float('y', 0.0, 10.0)
    return (x > 5) * 20 + y

def objective(trial: optuna.trial.Trial) -> float:
    c = trial.suggest_float("c", 0.0, 1.0)
    if c < 0.5:
        _x = trial.suggest_float('x', 0.0, 10.0)
        return random.uniform(0.5, 0.9)
    else:
        _y = trial.suggest_float('y', 0.5, 1.0)
        return random.uniform(0.8, 1.0)



sampler = optuna.samplers.RandomSampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=100)

# evaluator = PedAnovaImportanceEvaluator()
evaluator = ConditionalPedAnovaImportanceEvaluator()
fig = plot_param_importances(study, evaluator=evaluator)
fig.show()



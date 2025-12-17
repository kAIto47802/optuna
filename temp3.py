import optuna
from optuna.visualization import plot_param_importances
from optuna.importance._ped_anova.evaluator import PedAnovaImportanceEvaluator as ConditionalPedAnovaImportanceEvaluator
from optuna.importance._ped_anova.orig_evaluator import PedAnovaImportanceEvaluator

def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float('x', -5.0, 5.0)
    # w = trial.suggest_float('w', -1.0, 1.0)
    w = 1
    # c = trial.suggest_categorical('c', [True, False])
    c = trial.suggest_float("c", 0, 1)
    if c> 0.5:
        y = trial.suggest_float('y', -10.0, 10.0)
        # z = trial.suggest_float('z', -1.0, 1.0)
        z = 0
        return x - (y < -9) * 100
    return x


# def objective(trial: optuna.trial.Trial) -> float:
#     x = trial.suggest_float('x', -10.0, -10.0)
#     # w = trial.suggest_float('w', -1.0, 1.0)
#     c = trial.suggest_categorical('c', [True, False])
#     if c:
#         y = trial.suggest_float('y', 5.0, 10.0)
#         z = trial.suggest_float('z', 0.0, 1.0)
#         return x + y * 100 + z * 0.001
#     return x

sampler = optuna.samplers.RandomSampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100)

# evaluator = PedAnovaImportanceEvaluator()
evaluator = ConditionalPedAnovaImportanceEvaluator()
fig = plot_param_importances(study, evaluator=evaluator)
fig.show()
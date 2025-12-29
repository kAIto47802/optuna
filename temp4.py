import optuna
from optuna.visualization import plot_param_importances
import matplotlib.pyplot as plt
import numpy as np
import importlib

evaluator_name = "evaluator0b"

# evaluator0: regimeをパラメータごとに定義、alpha, beta使用
# evaluator0b: within-regime divergenceの係数にはalpha, betaを使用、inter-regime divergenceには使用しない
# evaluator0c: within-regime divergenceではalpha, betaを使用しない、inter-regime divergenceには使用する
# evaluator0d: within-regime, inter-regime divergenceともにalpha, betaを使用しない (evaluator2と等価)
# evaluator1: regimeをパラメータ共通で定義、(alpha, betaは使用するが、全パラメータ同じになる)
# evaluator2: そのパラメータが存在するtrialだけ使用
# evaluator: 通常のped-anova

PedAnovaImportanceEvaluator = importlib.import_module(f"optuna.importance._ped_anova.{evaluator_name}").PedAnovaImportanceEvaluator




def objective(trial: optuna.trial.Trial) -> float:
    c = trial.suggest_float("c", 0.0, 1.0)
    x = trial.suggest_float('x', -5.0, -4.0)
    y = trial.suggest_float('y', 4.0, 5.0)
    if c < 0.5:
        return x
    else:
        return y


def objective(trial: optuna.trial.Trial) -> float:
    c = trial.suggest_float("c", 0.0, 1.0)
    if c < 0.5:
        x = trial.suggest_float('x', -5.0, 2.0)
        return x
    else:
        y = trial.suggest_float('y', -2.0, 5.0)
        return y


sampler = optuna.samplers.RandomSampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=500)

region_quantile = 1.0
xs = np.arange(0.01, region_quantile - 0.01, 0.01)
normalized_importances = {"c": [], "x": [], "y": []}
importances = {"c": [], "x": [], "y": []}
for target_quantile in xs:
    evaluator = PedAnovaImportanceEvaluator(target_quantile=target_quantile, region_quantile=region_quantile)
    normalized_importance = optuna.importance.get_param_importances(study, evaluator=evaluator)
    importance = optuna.importance.get_param_importances(study, evaluator=evaluator, normalize=False)
    for param_name in ["c", "x", "y"]:
        importances[param_name].append(importance[param_name])
        normalized_importances[param_name].append(normalized_importance[param_name])

plt.figure(figsize=(8, 4))
plt.title(f"Parameter Importances vs Target Quantile (Region Quantile={region_quantile})")
plt.xlabel("Target Quantile")
plt.ylabel("Parameter Importance")
for param_name, values in importances.items():
    plt.plot(xs, values, label=f"param: {param_name}")
plt.legend()
plt.grid()

plt.savefig(f"{evaluator_name}-result.png")

plt.figure(figsize=(8, 4))
plt.title(f"Normalized Parameter Importances vs Target Quantile (Region Quantile={region_quantile})")
plt.xlabel("Target Quantile")
plt.ylabel("Normalized Parameter Importance")
for param_name, values in normalized_importances.items():
    plt.plot(xs, values, label=f"param: {param_name}")
plt.legend()
plt.grid()
plt.savefig(f"{evaluator_name}-result-normalized.png")
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
def experiments(n_seeds: int) -> dict[str, np.ndarray]:
    data = {"tpe-mv": [], "tpe-uv": [], "gp": []}
    for seed in range(n_seeds):
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed, constraints_func=constraints)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=N_TRIALS)
        data["tpe-mv"].append([t.value if t.user_attrs["c"] <= 0 else np.inf for t in study.trials])
        sampler = optuna.samplers.TPESampler(seed=seed, constraints_func=constraints)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=N_TRIALS)
        data["tpe-uv"].append([t.value if t.user_attrs["c"] <= 0 else np.inf for t in study.trials])
        sampler = optuna.samplers.GPSampler(seed=seed, constraints_func=constraints)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=N_TRIALS)
        data["gp"].append([t.value if t.user_attrs["c"] <= 0 else np.inf for t in study.trials])
    return {k: np.asarray(v) for k, v in data.items()}
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
data = experiments(n_seeds=10)
dx = np.arange(N_TRIALS) + 1
fig, ax = plt.subplots(figsize=(10, 5))
lines = []
labels = []
LABEL_DICT = {"tpe-mv": "Multivariate TPE", "tpe-uv": "TPE", "gp": "GP"}
COLOR_DICT = {"tpe-mv": "blue", "tpe-uv": "black", "gp": "darkred"}
for sampler_name, _values in data.items():
    if len(_values) == 0:
        continue
    color = COLOR_DICT[sampler_name]
    labels.append(LABEL_DICT[sampler_name])
    values = np.minimum.accumulate(_values, axis=-1)
    q75, meds, q25 = np.percentile(values, [75, 50, 25], axis=0)
    line, = ax.plot(dx, meds, color=color)
    lines.append(line)
    ax.fill_between(dx, q25, q75, color=color, alpha=0.2)
ax.set_xlim(1, N_TRIALS)
ax.set_ylim(-1, 6.5)
ax.set_xlabel("Number of Trials")
ax.set_ylabel("Feasible Objective Value")
ax.grid(which="minor", color="gray", linestyle=":")
ax.grid(which="major", color="black")
fig.legend(
    handles=lines,
    loc="lower center",
    labels=labels,
    bbox_to_anchor=(0.5, -0.2),
    fontsize=24,
    fancybox=False,
    ncol=len(lines),
)
plt.savefig("constraints2.png", bbox_inches="tight")
import matplotlib.pyplot as plt
import numpy as np
import optuna

import optuna_integration

N_TRIALS = 100
N_SEEDS = 10


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 0.0, 2 * np.pi)
    y = trial.suggest_float("y", 0.0, 2 * np.pi)
    return float(np.sin(x) + y)


def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    x = trial.params["x"]
    y = trial.params["y"]
    c = float(np.sin(x) * np.sin(y) + 0.95)
    trial.set_user_attr("c", c)
    return (c,)


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


def _extract_elapsed_time(study: optuna.study.Study) -> list[float]:
    return [
        (t.datetime_complete - study.trials[0].datetime_start).total_seconds()
        for t in study.trials
    ]


def experiments(n_seeds: int) -> dict[str, np.ndarray]:
    data = {"botorch": [], "gp": []}
    for seed in range(n_seeds):
        sampler = optuna_integration.BoTorchSampler(seed=seed, constraints_func=constraints)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=N_TRIALS)
        data["botorch"].append(_extract_elapsed_time(study))

        sampler = optuna.samplers.GPSampler(seed=seed, constraints_func=constraints)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=N_TRIALS)
        data["gp"].append(_extract_elapsed_time(study))

    return {k: np.asarray(v) for k, v in data.items()}


optuna.logging.set_verbosity(optuna.logging.CRITICAL)
data = experiments(n_seeds=N_SEEDS)

dx = np.arange(N_TRIALS) + 1
fig, ax = plt.subplots(figsize=(10, 5))
lines = []
labels = []
LABEL_DICT = {"botorch": "BoTorch", "gp": "GP"}
COLOR_DICT = {"botorch": "blue", "gp": "darkred"}

for sampler_name, values in data.items():
    if len(values) == 0:
        continue

    color = COLOR_DICT[sampler_name]
    labels.append(LABEL_DICT[sampler_name])

    # values = np.minimum.accumulate(_values, axis=-1)
    # q75, meds, q25 = np.percentile(values, [75, 50, 25], axis=0)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    (line,) = ax.plot(dx, mean, color=color)
    lines.append(line)
    ax.fill_between(
        dx, mean - std / np.sqrt(N_SEEDS), mean + std / np.sqrt(N_SEEDS), color=color, alpha=0.2
    )

ax.set_xlim(1, N_TRIALS)
ax.set_ylim(0, 180)
ax.set_xlabel("Number of Trials", fontsize=16)
ax.set_ylabel("Elapsed Time [s]", fontsize=16)
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.grid(which="minor", color="gray", linestyle=":")
ax.grid(which="major", color="black")
fig.legend(
    handles=lines,
    loc="lower center",
    labels=labels,
    bbox_to_anchor=(0.5, -0.1),
    fontsize=16,
    fancybox=False,
    ncol=len(lines),
)
plt.savefig("times3.png", bbox_inches="tight")

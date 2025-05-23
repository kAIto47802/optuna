import itertools

import numpy as np
import pandas as pd


df = pd.read_json("hvi-benchmarking-results.json")
for type_name in ["concave", "convex", "linear"]:
    instance_name = f"_generate_{type_name}_instances"
    df_target = df[df["instance_name"] == instance_name].drop(columns=["instance_name", "rtol"])

    print("<details>")
    print(f"<summary>Benchmark Results of {type_name.capitalize()}</summary>")
    print()
    print("|n_objectives,n_trials,n_calls| Box Decomposition (ms) | WFG (ms) | Speedup (x) |")
    print("|:--:|:--:|:--:|:--:|")
    for n_objectives, n_trials, n_calls in itertools.product(
        *(df["n_objectives"].unique(), df["n_trials"].unique(), df["n_calls"].unique())
    ):
        df_filtered = df_target[
            (df_target["n_objectives"] == n_objectives)
            & (df_target["n_trials"] == n_trials)
            & (df_target["n_calls"] == n_calls)
        ].drop(columns=["n_objectives", "n_trials", "n_calls"])
        ts_wfg = df_filtered["runtime_wfg"]
        ts_bd = df_filtered["runtime_bd"]
        print(
            f"| {n_objectives}, {n_trials}, {n_calls} "
            f"| {ts_bd.mean():.3f} $\pm$ {ts_bd.std() / np.sqrt(5):.3f} "
            f"| {ts_wfg.mean():.3f} $\pm$ {ts_wfg.std() / np.sqrt(5):.3f} "
            f"| {ts_wfg.mean() / ts_bd.mean():.3f} |"
        )
    print("</details>")
    print()

print(df["rtol"].unique())

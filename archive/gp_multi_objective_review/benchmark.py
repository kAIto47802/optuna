from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from optuna._hypervolume.box_decomposition import get_non_dominated_box_bounds
from optuna._hypervolume.box_decomposition2 import get_non_dominated_box_bounds2

# from optuna._hypervolume.wfg import compute_hypervolume
from optuna.study._multi_objective import _is_pareto_front


_EPS = 1e-12


class InstanceGenerator(Protocol):
    def __call__(self, n_trials: int, n_objectives: int, seed: int) -> np.ndarray:
        raise NotImplementedError


def _extract_pareto_sols(loss_values: np.ndarray) -> np.ndarray:
    sorted_loss_vals = np.unique(loss_values, axis=0)
    on_front = _is_pareto_front(sorted_loss_vals, assume_unique_lexsorted=True)
    return sorted_loss_vals[on_front]


def _generate_uniform_samples(n_trials: int, n_objectives: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.maximum(_EPS, rng.random((n_trials, n_objectives)))


def _generate_concave_instances(n_trials: int, n_objectives: int, seed: int) -> np.ndarray:
    """See Section 4.2 of https://arxiv.org/pdf/1510.01963"""
    points = _generate_uniform_samples(n_trials, n_objectives, seed)
    loss_values = points / np.maximum(_EPS, np.sum(points**2, axis=-1))[:, np.newaxis]
    return loss_values


def _generate_convex_instances(n_trials: int, n_objectives: int, seed: int) -> np.ndarray:
    """See Section 4.2 of https://arxiv.org/pdf/1510.01963"""
    points = _generate_uniform_samples(n_trials, n_objectives, seed)
    loss_values = 1 - points / np.maximum(_EPS, np.sum(points**2, axis=-1))[:, np.newaxis]
    return loss_values


def _generate_linear_instances(n_trials: int, n_objectives: int, seed: int) -> np.ndarray:
    """See Section 4.2 of https://arxiv.org/pdf/1510.01963"""
    points = _generate_uniform_samples(n_trials, n_objectives, seed)
    loss_values = points / np.maximum(_EPS, np.sum(points, axis=-1))[:, np.newaxis]
    return loss_values


def _calculate_hypervolume_improvement(
    lbs: np.ndarray, ubs: np.ndarray, new_sols: np.ndarray
) -> np.ndarray:
    diff = np.maximum(0.0, ubs - np.maximum(new_sols[..., np.newaxis, :], lbs))
    # The minimization version of Eq. (1) in https://arxiv.org/pdf/2006.05078.
    return np.sum(np.prod(diff, axis=-1), axis=-1)


# def benchmark_wfg(
#     pareto_sols: np.ndarray, new_sols: np.ndarray, ref_point: np.ndarray
# ) -> np.ndarray:
#     hv = compute_hypervolume(pareto_sols, ref_point, assume_pareto=True)
#     hvis = np.empty(len(new_sols))
#     for i, new_sol in enumerate(new_sols):
#         sols = np.vstack([pareto_sols, new_sol[np.newaxis, :]])
#         hvis[i] = compute_hypervolume(sols, ref_point, assume_pareto=False) - hv

#     return hvis


def benchmark_box_decomposition(
    pareto_sols: np.ndarray, new_sols: np.ndarray, ref_point: np.ndarray
) -> np.ndarray:
    lbs, ubs = get_non_dominated_box_bounds(pareto_sols, ref_point)
    return _calculate_hypervolume_improvement(lbs, ubs, new_sols)


def benchmark_box_decomposition2(
    pareto_sols: np.ndarray, new_sols: np.ndarray, ref_point: np.ndarray
) -> np.ndarray:
    lbs, ubs = get_non_dominated_box_bounds2(pareto_sols, ref_point)
    return _calculate_hypervolume_improvement(lbs, ubs, new_sols)


def main(
    gen: InstanceGenerator, n_objectives: int, n_trials: int, n_calls: int, seed: int
) -> tuple[float, float, float]:
    pareto_sols = _extract_pareto_sols(
        gen(n_trials=n_trials, n_objectives=n_objectives, seed=seed)
    )
    new_sols = gen(n_trials=n_calls, n_objectives=n_objectives, seed=seed + 100)
    ref_point = np.max(np.vstack([pareto_sols, new_sols]), axis=0)
    ref_point = np.maximum(ref_point * 0.9, ref_point * 1.1)

    print("Start")
    start = time.time()
    out1 = benchmark_box_decomposition(pareto_sols, new_sols, ref_point)
    time_box_decomp = (time.time() - start) * 1000
    print(f"Box Decomp.: {time_box_decomp:.3e} [ms]")

    start = time.time()
    out2 = benchmark_box_decomposition2(pareto_sols, new_sols, ref_point)
    # out2 = benchmark_wfg(pareto_sols, new_sols, ref_point)
    print(f"WFG: {(time.time() - start)*1000:.3e} [ms]")
    time_wfg = (time.time() - start) * 1000

    rtol = 1e-5  # 1e-5 is the default rtol in np.allclose.
    while not np.allclose(out1, out2, rtol=rtol):
        rtol *= 10.0

    return time_box_decomp, time_wfg, rtol


if __name__ == "__main__":
    import itertools

    import pandas as pd

    rows = []
    for gen, n_objectives, n_trials, n_calls, seed in itertools.product(
        *(
            [_generate_concave_instances, _generate_convex_instances, _generate_linear_instances],
            # range(2, 5),
            [20, 30],
            [10, 30, 100],
            [1000, 3000, 10000],
            range(5),
        )
    ):
        print(gen.__name__, f"{n_objectives=}, {n_trials=}, {n_calls=}, {seed=}")
        t_bd, t_wfg, rtol = main(gen, n_objectives, n_trials, n_calls, seed=seed)
        row = {
            "instance_name": gen.__name__,
            "n_objectives": n_objectives,
            "n_trials": n_trials,
            "n_calls": n_calls,
            "runtime_wfg": t_wfg,
            "runtime_bd": t_bd,
            "rtol": rtol,
            "seed": seed,
        }
        rows.append(row)

    pd.DataFrame(rows).to_json("hvi-benchmarking-results2.json")

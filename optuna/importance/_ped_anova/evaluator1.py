from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import cast

import numpy as np

from optuna._deprecated import _DEPRECATION_WARNING_TEMPLATE
from optuna._experimental import experimental_class
from optuna._warnings import optuna_warn
from optuna.distributions import BaseDistribution
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._ped_anova.scott_parzen_estimator import _build_parzen_estimator
from optuna.logging import get_logger
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = get_logger(__name__)


np.set_printoptions(precision=3)

class _QuantileFilter:
    def __init__(
        self,
        quantile: float,
        is_lower_better: bool,
        min_n_top_trials: int,
        target: Callable[[FrozenTrial], float] | None,
    ) -> None:
        assert 0 < quantile <= 1, "quantile must be in [0, 1]."
        assert min_n_top_trials > 0, "min_n_top_trials must be positive."

        self._quantile = quantile
        self._is_lower_better = is_lower_better
        self._min_n_top_trials = min_n_top_trials
        self._target = target

    def filter(self, trials: list[FrozenTrial]) -> list[FrozenTrial]:
        target, min_n_top_trials = self._target, self._min_n_top_trials
        sign = 1.0 if self._is_lower_better else -1.0
        loss_values = sign * np.asarray([t.value if target is None else target(t) for t in trials])
        err_msg = "len(trials) must be larger than or equal to min_n_top_trials"
        assert min_n_top_trials <= loss_values.size, err_msg

        def _quantile(v: np.ndarray, q: float) -> float:
            cutoff_index = int(np.ceil(q * loss_values.size)) - 1
            return float(np.partition(loss_values, cutoff_index)[cutoff_index])

        cutoff_val = max(
            np.partition(loss_values, min_n_top_trials - 1)[min_n_top_trials - 1],
            # TODO(nabenabe0928): After dropping Python3.10, replace below with
            # np.quantile(loss_values, self._quantile, method="inverted_cdf").
            _quantile(loss_values, self._quantile),
        )
        should_keep_trials = loss_values <= cutoff_val
        return [t for t, should_keep in zip(trials, should_keep_trials) if should_keep]


@experimental_class("3.6.0")
class PedAnovaImportanceEvaluator(BaseImportanceEvaluator):
    """PED-ANOVA importance evaluator.

    Implements the PED-ANOVA hyperparameter importance evaluation algorithm.

    PED-ANOVA fits Parzen estimators of :class:`~optuna.trial.TrialState.COMPLETE` trials better
    than a user-specified `target_quantile`.
    The importance can be interpreted as how important each hyperparameter is to get
    the performance better than `target_quantile`.

    For further information about PED-ANOVA algorithm, please refer to the following paper:

    - `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
      <https://arxiv.org/abs/2304.10255>`__

    `target_quantile` and `region_quantile` correspond to the parameters ``gamma'`` and ``gamma``
    in the original paper, respectively.

    .. note::

        The performance of PED-ANOVA depends on how many trials to consider above
        `target_quantile`. To stabilize the analysis, it is preferable to include at least
        5 trials above `target_quantile`.

    .. note::

        Please refer to `the original work <https://github.com/nabenabe0928/local-anova>`__.

    Args:
        target_quantile:
            Compute the importance of achieving top-``target_quantile`` quantile objective value.
            For example, ``target_quantile=0.1`` means that the importances give the information
            of which parameters were important to achieve the top-10% performance during
            optimization.

        region_quantile:
            Define the region where we compute the importance. For example,
            ``region_quantile=0.5`` means that we compute the importance in the region where
            trials achieve top-50% performance. If ``region_quantile=1.0``, the importance is
            computed in the whole search space.

        baseline_quantile:
            Compute the importance of achieving top-``baseline_quantile`` quantile objective value.
            For example, ``baseline_quantile=0.1`` means that the importances give the information
            of which parameters were important to achieve the top-10% performance during
            optimization.

            .. warning::
                Deprecated in v4.7.0. This feature will be removed in the future. The removal of
                this feature is currently scheduled for v0.6.0, but this schedule is subject to
                change. `baseline_quantile` is currently ignored. Use `target_quantile` instead.
                See https://github.com/optuna/optuna/releases/tag/v4.7.0.

        evaluate_on_local:
            Whether we measure the importance in the local or global space.
            If :obj:`True`, the importances imply how importance each parameter is during
            optimization. Meanwhile, ``evaluate_on_local=False`` gives the importances in the
            specified search_space. ``evaluate_on_local=True`` is especially useful when users
            modify search space during optimization.

    Example:
        An example of using PED-ANOVA is as follows:

        .. testcode::

            import optuna
            from optuna.importance import PedAnovaImportanceEvaluator


            def objective(trial):
                x1 = trial.suggest_float("x1", -10, 10)
                x2 = trial.suggest_float("x2", -10, 10)
                return x1 + x2 / 1000


            study = optuna.create_study()
            study.optimize(objective, n_trials=100)
            evaluator = PedAnovaImportanceEvaluator()
            importance = optuna.importance.get_param_importances(study, evaluator=evaluator)

    """

    def __init__(
        self,
        *,
        target_quantile: float = 0.1,  # gamma' in the original paper
        region_quantile: float = 1.0,  # gamma in the original paper
        baseline_quantile: float | None = None,
        evaluate_on_local: bool = True,
    ) -> None:
        assert 0.0 < target_quantile < region_quantile <= 1.0, (
            "condition 0.0 < `target_quantile` < `region_quantile` <= 1.0 must be satisfied"
        )
        if baseline_quantile is not None:
            msg = _DEPRECATION_WARNING_TEMPLATE.format(
                name="`baseline_quantile`", d_ver="4.7.0", r_ver="6.0.0"
            )
            optuna_warn(
                f"{msg} `baseline_quantile` is currently ignored. Use `target_quantile` instead.",
            )
        if region_quantile != 1.0 and not evaluate_on_local:
            optuna_warn("If `evaluate_on_local` is False, `region_quantile` has no effect.")

        self._target_quantile = target_quantile
        self._region_quantile = region_quantile
        self._evaluate_on_local = evaluate_on_local

        # Advanced Setups.
        # Discretize a domain [low, high] as `np.linspace(low, high, n_steps)`.
        self._n_steps: int = 50
        # Control the regularization effect by prior.
        self._prior_weight = 1.0
        # How many `trials` must be included in `top_trials`.
        self._min_n_top_trials = 2

    def _get_top_quantile_trials(
        self,
        study: Study,
        trials: list[FrozenTrial],
        quantile: float,
        target: Callable[[FrozenTrial], float] | None,
    ) -> list[FrozenTrial]:
        is_lower_better = study.directions[0] == StudyDirection.MINIMIZE
        if target is not None:
            optuna_warn(
                f"{self.__class__.__name__} computes the importances of params to achieve "
                "low `target` values. If this is not what you want, "
                "please modify target, e.g., by multiplying the output by -1."
            )
            is_lower_better = True

        top_trials = _QuantileFilter(
            quantile, is_lower_better, self._min_n_top_trials, target
        ).filter(trials)

        return top_trials

    def _compute_pearson_divergence(
        self,
        param_name: str,
        dist: BaseDistribution,
        target_trials: list[FrozenTrial],
        region_trials: list[FrozenTrial],
    ) -> float:
        # When pdf_all == pdf_top, i.e. all_trials == top_trials, this method will give 0.0.
        prior_weight = self._prior_weight
        pe_top = _build_parzen_estimator(
            param_name, dist, target_trials, self._n_steps, prior_weight
        )
        # NOTE: pe_top.n_steps could be different from self._n_steps.
        grids = np.arange(pe_top.n_steps)
        pdf_top = pe_top.pdf(grids) + 1e-12

        if self._evaluate_on_local:  # The importance of param during the study.
            pe_local = _build_parzen_estimator(
                param_name, dist, region_trials, self._n_steps, prior_weight
            )
            pdf_local = pe_local.pdf(grids) + 1e-12
        else:  # The importance of param in the search space.
            pdf_local = np.full(pe_top.n_steps, 1.0 / pe_top.n_steps)

        return float(pdf_local @ ((pdf_top / pdf_local - 1) ** 2))

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        all_dists = _get_distributions(study, params=params)
        if params is None:
            params = list({k for d in all_dists for k in d})

        assert params is not None

        trials = _get_filtered_trials(study, target=target)
        print("total trials: ", len(trials))
        # assert False
        # The following should be tested at _get_filtered_trials.
        assert target is not None or max([len(t.values) for t in trials], default=1) == 1
        if len(trials) <= self._min_n_top_trials:
            return {k: 0.0 for k in params}

        target_trials = self._get_top_quantile_trials(study, trials, self._target_quantile, target)
        region_trials = (
            trials
            if self._region_quantile == 1.0
            else self._get_top_quantile_trials(study, trials, self._region_quantile, target)
        )
        print("region_trials: ", len(region_trials))
        print("target_trials: ", len(target_trials))
        quantile = len(target_trials) / len(region_trials)  # gamma' / gamma
        param_importances: dict[str, float] = defaultdict(float)
        print("params:", params)
        regime_trials_map = _partition_by_regime(region_trials, target_trials)
        for param_name in params:
            print(f"===[[{param_name}]]" + "=" * 30)
            for dists, region_trials_regime, target_trials_regime in regime_trials_map:
                dist = dists.get(param_name)
                print(f"---{dist}: {len(region_trials_regime)} trials" + "-" * 30)
                print(f">> target trials: {len(target_trials_regime)}")
                regime_prob_target = len(target_trials_regime) / len(target_trials)  # a_i
                regime_prob_region = len(region_trials_regime) / len(region_trials)  # b_i
                print(f"regime_prob_target: {regime_prob_target}")
                print(f"regime_prob_region: {regime_prob_region}")

                print("target_trials_regime:")
                print("> value: ", np.array([t.value for t in target_trials_regime]))
                print("> param: ", np.array([t.params.get(param_name) for t in target_trials_regime]))
                print("region_trials_regime:")
                print("> value: ", np.array([t.value for t in region_trials_regime]))
                print("> param: ", np.array([t.params.get(param_name) for t in region_trials_regime]))
                if dist is not None and not dist.single() and len(target_trials_regime):
                    # between-regime divergence
                    param_importances[param_name] += (
                        tmp := regime_prob_target**2
                        / regime_prob_region
                        * self._compute_pearson_divergence(
                            param_name,
                            dist,
                            target_trials=target_trials_regime,
                            region_trials=region_trials_regime,
                        )
                    )
                    print(f"contribution from within-regime pearson divergence: {tmp}")
                else:
                    print("contribution from within-regime pearson divergence: (0.0)")
                # inter-regime divergence
                param_importances[param_name] += (
                    tmp2 := (regime_prob_target - regime_prob_region) ** 2 / regime_prob_region
                )
                print(f"contribution from inter-regime divergence: {tmp2}")
        param_importances = {k: v * quantile**2 for k, v in param_importances.items()}
        return _sort_dict_by_importance(param_importances)


def _partition_by_regime(
    region_trials: list[FrozenTrial], target_trials: list[FrozenTrial]
) -> list[tuple[dict[str, BaseDistribution], list[FrozenTrial], list[FrozenTrial]]]:
    #!!! Dynamic range is not supported
    regime_id_trial_map: dict[int, tuple[list[FrozenTrial], list[FrozenTrial]]] = defaultdict(lambda: ([], []))
    regimes: list[dict[str, BaseDistribution]] = []
    all_target_trial_ids = set([t._trial_id for t in target_trials])
    for trial in region_trials:
        regime_id = next((i for i, r in enumerate(regimes) if r == trial.distributions), None)
        if regime_id is None:
            regime_id = len(regimes)
            regimes.append(trial.distributions)
        regime_id_trial_map[regime_id][0].append(trial)
        if trial._trial_id in all_target_trial_ids:
            regime_id_trial_map[regime_id][1].append(trial)
    return [
        (regimes[regime_id], region_trials_regime, target_trials_regime)
        for regime_id, (region_trials_regime, target_trials_regime) in regime_id_trial_map.items()
    ]


def _get_filtered_trials(
    study: Study, target: Callable[[FrozenTrial], float] | None
) -> list[FrozenTrial]:
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return [
        trial
        for trial in trials
        if np.isfinite(
            target(trial) if target is not None else cast("float", trial.value)
        )  # TC006
    ]


def _get_distributions(
    study: Study, params: list[str] | None
) -> list[dict[str, BaseDistribution]]:
    if params is not None:
        raise NotImplementedError()
    trials = study.get_trials(deepcopy=False)
    return [
        t.distributions
        for t in trials
        if t.state
        in (
            TrialState.COMPLETE,
            TrialState.WAITING,
            TrialState.RUNNING,
        )
    ]

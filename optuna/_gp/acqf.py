from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING

import numpy as np

from optuna._gp.gp import kernel
from optuna._gp.gp import KernelParamsTensor
from optuna._gp.gp import posterior
from optuna._gp.search_space import ScaleType
from optuna._gp.search_space import SearchSpace


if TYPE_CHECKING:
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


class BaseAcquisitionFunction(abc.ABC):
    def __init__(
        self,
        kernel_params: KernelParamsTensor,
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
    ):
        self.kernel_params = kernel_params
        self.search_space = search_space
        self.X = X

        X_tensor = torch.from_numpy(X)
        is_categorical = torch.from_numpy(search_space.scale_types == ScaleType.CATEGORICAL)
        with torch.no_grad():
            cov_Y_Y = kernel(is_categorical, kernel_params, X_tensor, X_tensor).detach().numpy()
        cov_Y_Y[np.diag_indices(X.shape[0])] += kernel_params.noise_var.item()
        cov_Y_Y_inv = np.linalg.inv(cov_Y_Y)

        self.cov_Y_Y_inv = cov_Y_Y_inv
        self.cov_Y_Y_inv_Y = cov_Y_Y_inv @ Y

    @abc.abstractmethod
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def eval_no_grad(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.eval(torch.from_numpy(x)).detach().numpy()

    def eval_with_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        assert x.ndim == 1
        x_tensor = torch.from_numpy(x)
        x_tensor.requires_grad_(True)
        val = self.eval(x_tensor)
        val.backward()  # type: ignore
        return val.item(), x_tensor.grad.detach().numpy()  # type: ignore

    def _get_mean_var(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return posterior(
            self.kernel_params,
            torch.from_numpy(self.X),
            torch.from_numpy(self.search_space.scale_types == ScaleType.CATEGORICAL),
            torch.from_numpy(self.cov_Y_Y_inv),
            torch.from_numpy(self.cov_Y_Y_inv_Y),
            x,
        )


class LogEI(BaseAcquisitionFunction):
    def __init__(
        self,
        kernel_params: KernelParamsTensor,
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
        acqf_stabilizing_noise: float = 1e-12,
    ):
        super().__init__(kernel_params, search_space, X, Y)
        self.max_Y = np.max(Y)
        self.acqf_stabilizing_noise = acqf_stabilizing_noise

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._get_mean_var(x)
        return logei(mean=mean, var=var + self.acqf_stabilizing_noise, f0=self.max_Y)


class BaseConfidenceBound(BaseAcquisitionFunction):
    def __init__(
        self,
        kernel_params: KernelParamsTensor,
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
        beta: float,
    ):
        super().__init__(kernel_params, search_space, X, Y)
        self.beta = beta


class UCB(BaseConfidenceBound):
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._get_mean_var(x)
        return ucb(mean=mean, var=var, beta=self.beta)


class LCB(BaseConfidenceBound):
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._get_mean_var(x)
        return lcb(mean=mean, var=var, beta=self.beta)


def standard_logei(z: torch.Tensor) -> torch.Tensor:
    # Return E_{x ~ N(0, 1)}[max(0, x+z)]

    # We switch the implementation depending on the value of z to
    # avoid numerical instability.
    small = z < -25

    vals = torch.empty_like(z)
    # Eq. (9) in ref: https://arxiv.org/pdf/2310.20708.pdf
    # NOTE: We do not use the third condition because ours is good enough.
    z_small = z[small]
    z_normal = z[~small]
    sqrt_2pi = math.sqrt(2 * math.pi)
    # First condition
    cdf = 0.5 * torch.special.erfc(-z_normal * math.sqrt(0.5))
    pdf = torch.exp(-0.5 * z_normal**2) * (1 / sqrt_2pi)
    vals[~small] = torch.log(z_normal * cdf + pdf)
    # Second condition
    r = math.sqrt(0.5 * math.pi) * torch.special.erfcx(-z_small * math.sqrt(0.5))
    vals[small] = -0.5 * z_small**2 + torch.log((z_small * r + 1) * (1 / sqrt_2pi))
    return vals


def logei(mean: torch.Tensor, var: torch.Tensor, f0: float) -> torch.Tensor:
    # Return E_{y ~ N(mean, var)}[max(0, y-f0)]
    sigma = torch.sqrt(var)
    st_val = standard_logei((mean - f0) / sigma)
    val = torch.log(sigma) + st_val
    return val


def ucb(mean: torch.Tensor, var: torch.Tensor, beta: float) -> torch.Tensor:
    return mean + torch.sqrt(beta * var)


def lcb(mean: torch.Tensor, var: torch.Tensor, beta: float) -> torch.Tensor:
    return mean - torch.sqrt(beta * var)

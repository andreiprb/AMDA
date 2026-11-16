import time
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, Matern, WhiteKernel

from .config import Config


class GP:
    @staticmethod
    def kernel(
        base: str,
        d: int,
        ard: bool,
        signal: float = 1.0,
        length_scale: float = 1.0,
        noise: float = 0.1,
    ) -> Kernel:
        ls = np.full(d, length_scale) if ard else length_scale
        if base == "rbf":
            k = RBF(ls, Config.LS_BOUNDS)
        elif base == "matern":
            k = Matern(ls, Config.LS_BOUNDS, nu=Config.MATERN_NU)
        else:
            raise ValueError(f"Unknown kernel: {base}")
        return (ConstantKernel(signal, Config.SIGNAL_BOUNDS) * k
                + WhiteKernel(noise, Config.NOISE_BOUNDS))

    @staticmethod
    def fit(
        X: np.ndarray,
        y: np.ndarray,
        kernel: Kernel,
        n_restarts: int,
    ) -> tuple[GaussianProcessRegressor, float, int]:
        gp = GaussianProcessRegressor(
            kernel,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=Config.RANDOM_STATE,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t = time.perf_counter()
            gp.fit(X, y)
            fit_time = time.perf_counter() - t
        opt_fail = sum("lbfgs" in str(w.message) for w in caught)
        return gp, fit_time, opt_fail

    @staticmethod
    def hyper(gp: GaussianProcessRegressor) -> dict:
        k = gp.kernel_
        return dict(
            lml=float(gp.log_marginal_likelihood_value_),
            n_hyper=len(k.theta),
            signal_var=float(k.k1.k1.constant_value),
            noise_var=float(k.k2.noise_level),
            length_scale=np.atleast_1d(k.k1.k2.length_scale).tolist(),
        )

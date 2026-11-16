import numpy as np
from scipy.stats import norm


class Metrics:
    @staticmethod
    def regression(
        y: np.ndarray,
        mu: np.ndarray,
        sd: np.ndarray,
        level: float,
    ) -> dict:
        r = y - mu
        z = norm.ppf(0.5 + level / 2)
        return dict(
            mae=float(np.mean(np.abs(r))),
            rmse=float(np.sqrt(np.mean(r ** 2))),
            r2=float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)),
            nlpd=float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + r ** 2 / (2 * sd ** 2))),
            coverage=float(np.mean(np.abs(r) <= z * sd)),
            width=float(np.mean(2 * z * sd)),
        )

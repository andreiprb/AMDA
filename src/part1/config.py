from pathlib import Path

import yaml


class _ConstMeta(type):
    def __setattr__(cls, name, value):
        raise AttributeError(
            f"Cannot set attribute on const class: {name}"
        )

    def __delattr__(cls, name):
        raise AttributeError(
            f"Cannot delete attribute on const class: {name}"
        )

    def __call__(cls, *args, **kwargs):
        raise TypeError(
            "Config is a static container; cannot be instantiated"
        )


class Config(metaclass=_ConstMeta):
    @staticmethod
    def _load(path: Path) -> dict:
        with path.open("r") as f:
            return yaml.safe_load(f) or {}

    ROOT:     Path = Path(__file__).resolve().parents[2]
    CFG_PATH: Path = ROOT / "config" / "config.yaml"
    CFG:      dict = _load.__func__(CFG_PATH)

    _P1: dict = CFG.get("part1") or {}

    RANDOM_STATE: int = int(_P1.get("random_state", 42))
    CV_FOLDS:     int = int(_P1.get("cv_folds",     5))

    _DATA: dict = _P1.get("data") or {}

    CACHE_DIR: Path      = ROOT / _DATA.get("cache_dir", "data/.cache")
    DATASETS:  list[str] = list(_DATA.get("datasets", [
        "concrete", "airfoil", "parkinsons", "qsar"
    ]))
    PARKINSONS_PER_SUBJECT: int = int(_DATA.get("parkinsons_per_subject", 0))

    _GP: dict = _P1.get("gp") or {}

    KERNELS:        list[str] = list(_GP.get("kernels", ["rbf", "matern"]))
    MATERN_NU:      float     = float(_GP.get("matern_nu",      1.5))
    N_RESTARTS_ISO: int       = int(_GP.get("n_restarts_iso", 5))
    N_RESTARTS_ARD: int       = int(_GP.get("n_restarts_ard", 2))

    SIGNAL_BOUNDS: tuple[float, float] = tuple(map(float, _GP.get("signal_bounds",       [1e-2, 1e2])))
    LS_BOUNDS:     tuple[float, float] = tuple(map(float, _GP.get("length_scale_bounds", [1e-2, 1e3])))
    NOISE_BOUNDS:  tuple[float, float] = tuple(map(float, _GP.get("noise_bounds",        [1e-6, 1e1])))

    _EVAL: dict = _P1.get("evaluation") or {}

    INTERVAL_LEVEL: float = float(_EVAL.get("interval_level", 0.95))
    RESULTS_DIR:    Path  = ROOT / _EVAL.get("results_dir", "results/part1")

import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config
from .datasets import Datasets
from .gp import GP
from .metrics import Metrics


class Experiment:
    KEY:        list[str] = ["dataset", "kernel", "ard", "fold"]
    RUNS_PATH:  Path      = Config.RESULTS_DIR / "runs.csv"
    PREDS_PATH: Path      = Config.RESULTS_DIR / "predictions.csv"

    @staticmethod
    def run(datasets: list[str] | None = None) -> None:
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        done = {
            (r["dataset"], r["kernel"], bool(r["ard"]), int(r["fold"]))
            for r in Experiment.runs().to_dict("records")
        }
        for name in datasets or Config.DATASETS:
            ds = Datasets.load(name)
            for fold, (tr, te) in enumerate(Datasets.folds(ds)):
                todo = [
                    (base, ard) for base in Config.KERNELS for ard in (False, True)
                    if (name, base, ard, fold) not in done
                ]
                if not todo:
                    continue
                scaler = StandardScaler().fit(ds.X[tr])
                split = dict(
                    Xtr=scaler.transform(ds.X[tr]), ytr=ds.y[tr],
                    Xte=scaler.transform(ds.X[te]), yte=ds.y[te], idx=te,
                )
                for base, ard in todo:
                    Experiment._fit(name, base, ard, fold, split)

    @staticmethod
    def runs() -> pd.DataFrame:
        if not Experiment.RUNS_PATH.exists():
            return pd.DataFrame(columns=Experiment.KEY)
        return pd.read_csv(Experiment.RUNS_PATH).drop_duplicates(Experiment.KEY, keep="last")

    @staticmethod
    def predictions() -> pd.DataFrame:
        if not Experiment.PREDS_PATH.exists():
            return pd.DataFrame(columns=Experiment.KEY + ["idx", "y", "mu", "sd"])
        return pd.read_csv(Experiment.PREDS_PATH).drop_duplicates(Experiment.KEY + ["idx"], keep="last")

    @staticmethod
    def status() -> pd.DataFrame:
        runs = Experiment.runs()
        counts = runs.groupby("dataset").size() if len(runs) else pd.Series(dtype=int)
        expected = Config.CV_FOLDS * len(Config.KERNELS) * 2
        return pd.DataFrame(
            {"done": [int(counts.get(d, 0)) for d in Config.DATASETS], "expected": expected},
            index=pd.Index(Config.DATASETS, name="dataset"),
        )

    @staticmethod
    def _fit(name: str, base: str, ard: bool, fold: int, split: dict) -> dict:
        Xtr, ytr, Xte, yte = split["Xtr"], split["ytr"], split["Xte"], split["yte"]
        kernel = GP.kernel(base, Xtr.shape[1], ard)
        n_restarts = Config.N_RESTARTS_ARD if ard else Config.N_RESTARTS_ISO
        gp, fit_time, opt_fail = GP.fit(Xtr, ytr, kernel, n_restarts)
        mu, sd = gp.predict(Xte, return_std=True)

        key = dict(dataset=name, kernel=base, ard=ard, fold=fold)
        Experiment._append(
            Experiment.PREDS_PATH,
            pd.DataFrame({**key, "idx": split["idx"], "y": yte, "mu": mu, "sd": sd}),
        )
        row = dict(
            **key, n_train=len(ytr), n_test=len(yte),
            **Metrics.regression(yte, mu, sd, Config.INTERVAL_LEVEL),
            **GP.hyper(gp),
            fit_time=fit_time, opt_fail=opt_fail,
        )
        row["length_scale"] = json.dumps(row["length_scale"])
        Experiment._append(Experiment.RUNS_PATH, pd.DataFrame([row]))

        print(
            f"{name:<11} fold {fold + 1}/{Config.CV_FOLDS}  {base + ('-ard' if ard else '-iso'):<10}"
            f"  rmse {row['rmse']:<8.4g}  nlpd {row['nlpd']:.3f}  cov {row['coverage']:.2f}"
            f"  lml {row['lml']:.1f}  {fit_time:.1f}s",
            flush=True,
        )
        return row

    @staticmethod
    def _append(path: Path, df: pd.DataFrame) -> None:
        df.to_csv(path, mode="a", header=not path.exists(), index=False)

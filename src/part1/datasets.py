import io
import os
import pickle
import urllib.request
import zipfile
from typing import NamedTuple

import certifi
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from ucimlrepo import fetch_ucirepo

from .config import Config

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = os.environ["SSL_CERT_FILE"]


class Dataset(NamedTuple):
    X:        np.ndarray
    y:        np.ndarray
    groups:   np.ndarray | None
    features: list[str]


class Datasets:
    _SPECS: dict[str, dict] = {
        "concrete": dict(
            uci_id=165,
            target="Concrete compressive strength",
            log=["Age"],
        ),
        "airfoil": dict(
            uci_id=291,
            target="scaled-sound-pressure",
            log=["frequency"],
        ),
        "parkinsons": dict(
            uci_id=189,
            target="motor_UPDRS",
            group="subject#",
            drop=["total_UPDRS", "test_time"],
        ),
        "qsar": dict(
            url="https://archive.ics.uci.edu/static/public/505/qsar+aquatic+toxicity.zip",
            file="qsar_aquatic_toxicity.csv",
            columns=["TPSA(Tot)", "SAacc", "H-050", "MLOGP",
                     "RDCHI", "GATS1p", "nN", "C-040", "LC50"],
            target="LC50",
        ),
    }

    @staticmethod
    def load(name: str) -> Dataset:
        if name not in Datasets._SPECS:
            raise ValueError(
                f"Unknown dataset: {name}; expected one of {list(Datasets._SPECS)}"
            )
        spec = Datasets._SPECS[name]
        df = Datasets._cached(name, lambda: Datasets._fetch(spec))
        if "group" in spec and Config.PARKINSONS_PER_SUBJECT:
            df = (df.groupby(spec["group"], group_keys=False)
                    .sample(n=Config.PARKINSONS_PER_SUBJECT, random_state=Config.RANDOM_STATE)
                    .reset_index(drop=True))

        y = df.pop(spec["target"]).to_numpy(np.float64)
        groups = df.pop(spec["group"]).to_numpy() if "group" in spec else None
        df = df.drop(columns=spec.get("drop", []))

        for col in spec.get("log", []):
            df[col] = np.log(df[col])
        df = df.rename(columns={c: f"log({c})" for c in spec.get("log", [])})

        return Dataset(
            X=np.ascontiguousarray(df.to_numpy(np.float64)),
            y=np.ascontiguousarray(y),
            groups=groups,
            features=df.columns.tolist(),
        )

    @staticmethod
    def folds(ds: Dataset) -> list[tuple[np.ndarray, np.ndarray]]:
        if ds.groups is not None:
            return list(GroupKFold(Config.CV_FOLDS).split(ds.X, ds.y, ds.groups))
        kf = KFold(Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        return list(kf.split(ds.X))

    @staticmethod
    def _fetch(spec: dict) -> pd.DataFrame:
        if "uci_id" in spec:
            return fetch_ucirepo(id=spec["uci_id"]).data.original

        with urllib.request.urlopen(spec["url"]) as r:
            archive = zipfile.ZipFile(io.BytesIO(r.read()))
        with archive.open(spec["file"]) as f:
            return pd.read_csv(f, sep=";", header=None, names=spec["columns"])

    @staticmethod
    def _cached(key: str, fetch_fn) -> pd.DataFrame:
        Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = Config.CACHE_DIR / f"{key}.pkl"
        if p.exists():
            with p.open("rb") as f:
                return pickle.load(f)
        df = fetch_fn()
        with p.open("wb") as f:
            pickle.dump(df, f)
        return df

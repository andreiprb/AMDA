import json
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, t as student_t
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from .config import Config
from .datasets import Datasets
from .gp import GP


class Analysis:
    FIELDS: dict[str, str] = {
        "concrete":   "civil engineering",
        "airfoil":    "aerospace acoustics",
        "parkinsons": "medicine",
        "qsar":       "chemistry",
    }
    TARGETS: dict[str, str] = {
        "concrete":   "compressive strength (MPa)",
        "airfoil":    "sound pressure (dB)",
        "parkinsons": "motor UPDRS",
        "qsar":       "LC50 (−log mol/L)",
    }
    LABELS: dict[str, str] = {
        "mae": "MAE", "rmse": "RMSE", "r2": "R²", "nlpd": "NLPD",
        "coverage": "coverage", "width": "interval width",
    }
    MODELS: list[str]      = ["SE iso", "SE ARD", "Matérn iso", "Matérn ARD"]
    # Hue encodes the kernel (blue SE, orange Matérn); shade and marker encode iso vs. ARD.
    COLORS: dict[str, str]  = dict(zip(MODELS, ["#5598e7", "#184f95", "#eb6834", "#9a3a12"]))
    MARKERS: dict[str, str] = dict(zip(MODELS, ["o", "D", "o", "D"]))
    ACCENT, INK, INK2, MUTED = "#2a78d6", "#0b0b0b", "#52514e", "#898781"
    AXIS, GRID, BAND        = "#c3c2b7", "#e1e0d9", "#f6f5f1"
    STYLE: dict[str, object] = {
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 12, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK2, "axes.labelsize": 12, "axes.labelpad": 6,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 2, "ytick.minor.size": 2,
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 12,
    }
    # Larger text for the figures that are shown as half-width subfigures.
    STYLE_SUB: dict[str, object] = {
        **STYLE, "font.size": 18, "axes.labelsize": 19, "axes.titlesize": 20,
        "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 18,
    }
    SHORT: dict[str, str] = {"suction-side-displacement-thickness": "displacement-thickness"}
    CRITERIA: dict[str, tuple[str, bool]] = {
        "rmse":    ("CV RMSE", True),
        "nlpd":    ("CV NLPD", True),
        "cal_err": ("CV calibration error", True),
        "lml":     ("max LML", False),
        "bic":     ("min BIC", True),
    }

    @staticmethod
    def overview() -> pd.DataFrame:
        rows = []
        for name in Config.DATASETS:
            ds = Datasets.load(name)
            rows.append(dict(
                dataset=name, field=Analysis.FIELDS[name],
                n=len(ds.y), d=ds.X.shape[1],
                groups=len(np.unique(ds.groups)) if ds.groups is not None else "–",
                target=Analysis.TARGETS[name],
                y_mean=ds.y.mean(), y_sd=ds.y.std(), y_min=ds.y.min(), y_max=ds.y.max(),
            ))
        return pd.DataFrame(rows).set_index("dataset")

    @staticmethod
    @plt.rc_context(STYLE)
    def plot_targets() -> None:
        names = Config.DATASETS
        fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 2.8), layout="constrained")
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            ax.hist(Datasets.load(name).y, bins=30, color=Analysis.ACCENT, edgecolor="white", linewidth=0.6)
            Analysis._title(ax, name, Analysis.FIELDS[name])
            ax.set_xlabel(Analysis.TARGETS[name])
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("count")
        Analysis._show(fig, "targets")

    @staticmethod
    def settings() -> pd.Series:
        return pd.Series({
            "CV folds":                     Config.CV_FOLDS,
            "random state":                 Config.RANDOM_STATE,
            "Matérn ν":                     Config.MATERN_NU,
            "restarts, iso":                Config.N_RESTARTS_ISO,
            "restarts, ARD":                Config.N_RESTARTS_ARD,
            "σ_f² bounds":                  Config.SIGNAL_BOUNDS,
            "ℓ bounds":                     Config.LS_BOUNDS,
            "σ_n² bounds":                  Config.NOISE_BOUNDS,
            "interval level":               Config.INTERVAL_LEVEL,
        }, name="value")

    @staticmethod
    def table(runs: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        g = Analysis._scores(runs).groupby(["dataset", "model"], observed=True)[cols]
        mean, sd = g.mean(), g.std()
        return pd.DataFrame(
            {c: [f"{m:.3g} ± {s:.2g}" for m, s in zip(mean[c], sd[c])] for c in cols},
            index=mean.index,
        )

    @staticmethod
    @plt.rc_context(STYLE_SUB)
    def plot_metrics(runs: pd.DataFrame, cols: list[str]) -> None:
        df = Analysis._scores(runs)
        names = list(df.dataset.cat.categories)
        fig, axes = plt.subplots(
            len(cols), len(names),
            figsize=(3.2 * len(names), 2.4 * len(cols)), squeeze=False, layout="constrained",
        )
        for i, c in enumerate(cols):
            for j, name in enumerate(names):
                ax = axes[i, j]
                g = df[df.dataset == name].groupby("model", observed=True)[c]
                mean, sd = g.mean(), g.std()
                for k, m in enumerate(mean.index):
                    ax.errorbar(
                        k, mean[m], yerr=sd[m], fmt=Analysis.MARKERS[m], ms=10, color=Analysis.COLORS[m],
                        mec="white", mew=1.1, elinewidth=2.4, capsize=6, capthick=2.4, zorder=3,
                    )
                label = Analysis.LABELS.get(c, c)
                if c == "coverage":
                    ax.axhline(Config.INTERVAL_LEVEL, color=Analysis.MUTED, lw=1.5, zorder=1)
                    label += f" (line: {Config.INTERVAL_LEVEL:.0%})"
                # Two-level tick labels on the bottom row: iso/ARD under each point, kernel under each pair.
                last = i == len(cols) - 1
                parts = [str(m).rsplit(" ", 1) for m in mean.index]
                ax.set_xticks(range(len(mean)), [v if last else "" for _, v in parts])
                if last:
                    for kern in dict.fromkeys(k for k, _ in parts):
                        ax.annotate(
                            kern, (np.mean([p for p, (k, _) in enumerate(parts) if k == kern]), 0),
                            xycoords=ax.get_xaxis_transform(), xytext=(0, -32), textcoords="offset points",
                            ha="center", va="top", fontsize=17, fontweight="bold",
                        )
                ax.set_xlim(-0.6, len(mean) - 0.4)
                ax.tick_params(axis="x", length=0)
                ax.set_title(name if i == 0 else "")
                ax.set_ylabel(label if j == 0 else "")
                ax.grid(axis="x", visible=False)
        fig.get_layout_engine().set(hspace=0.08)
        fig.align_ylabels(axes[:, 0])
        Analysis._show(fig, "metrics_" + "_".join(cols))

    @staticmethod
    @plt.rc_context(STYLE)
    def plot_calibration(preds: pd.DataFrame) -> None:
        df = Analysis._label(preds)
        names = list(df.dataset.cat.categories)
        levels = np.linspace(0.05, 0.99, 40)
        z = norm.ppf(0.5 + levels / 2)
        fig, axes = plt.subplots(
            1, len(names), figsize=(3.2 * len(names), 3.7), sharey=True, layout="constrained",
        )
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            ax.plot([0, 1], [0, 1], color=Analysis.MUTED, lw=1, zorder=1)
            for m, g in df[df.dataset == name].groupby("model", observed=True):
                r = np.abs(g.y - g.mu).to_numpy() / g.sd.to_numpy()
                ax.plot(levels, (r[:, None] <= z[None, :]).mean(axis=0), color=Analysis.COLORS[m])
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.set_title(name)
            ax.set_xlabel("nominal coverage")
        note = dict(fontsize=10, color=Analysis.MUTED, style="italic", transform=axes[0].transAxes)
        axes[0].text(0.05, 0.95, "intervals too wide", va="top", **note)
        axes[0].text(0.95, 0.05, "intervals too narrow", ha="right", **note)
        axes[0].set_ylabel("empirical coverage")
        Analysis._legend(fig, list(df.model.cat.categories), line=True)
        Analysis._show(fig, "calibration")

    @staticmethod
    def calibration_error(preds: pd.DataFrame) -> pd.DataFrame:
        levels = np.linspace(0.05, 0.95, 19)
        z = norm.ppf(0.5 + levels / 2)

        def err(g: pd.DataFrame) -> float:
            r = (np.abs(g.y - g.mu) / g.sd).to_numpy()
            return float(np.abs((r[:, None] <= z).mean(axis=0) - levels).mean())

        return (
            preds.groupby(["dataset", "kernel", "ard", "fold"])
            .apply(err, include_groups=False)
            .rename("cal_err")
            .reset_index()
        )

    @staticmethod
    @plt.rc_context(STYLE)
    def plot_residuals(preds: pd.DataFrame) -> None:
        df = Analysis._label(preds)
        names = list(df.dataset.cat.categories)
        fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.5), layout="constrained")
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            ax.plot([-4, 4], [-4, 4], color=Analysis.MUTED, lw=1, zorder=1)
            for m, g in df[df.dataset == name].groupby("model", observed=True):
                r = np.sort(((g.y - g.mu) / g.sd).to_numpy())
                q = norm.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
                ax.plot(q, r, Analysis.MARKERS[m], ms=2.6, mew=0, alpha=0.8, color=Analysis.COLORS[m])
            ax.set_title(name)
            ax.set_xlabel("normal quantile")
        axes[0].set_ylabel("standardized residual")
        Analysis._legend(fig, list(df.model.cat.categories))
        Analysis._show(fig, "residuals")

    @staticmethod
    def residual_stats(preds: pd.DataFrame) -> pd.DataFrame:
        df = Analysis._label(preds)
        df["z"] = (df.y - df.mu) / df.sd
        return df.groupby(["dataset", "model"], observed=True).z.agg(
            sd="std",
            excess_kurtosis=lambda s: kurtosis(s),
            beyond_2sd=lambda s: (s.abs() > 2).mean(),
        )

    @staticmethod
    def selection(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = Analysis._scores(runs)
        grouped = df.groupby(["dataset", "fold"], observed=True)

        ranks = pd.DataFrame({
            label: grouped[c].rank(ascending=asc)
            for c, (label, asc) in Analysis.CRITERIA.items()
        })
        ranks[["dataset", "model"]] = df[["dataset", "model"]]
        mean_rank = ranks.groupby(["dataset", "model"], observed=True).mean().round(2)

        best = {}
        for c, (_, asc) in Analysis.CRITERIA.items():
            idx = grouped[c].idxmin() if asc else grouped[c].idxmax()
            best[c] = pd.Series(df.loc[idx.to_numpy(), "model"].astype(str).to_numpy(), index=idx.index)
        agreement = pd.DataFrame({
            "LML = CV RMSE": best["lml"] == best["rmse"],
            "LML = CV NLPD": best["lml"] == best["nlpd"],
            "BIC = CV RMSE": best["bic"] == best["rmse"],
            "BIC = CV NLPD": best["bic"] == best["nlpd"],
        }).groupby(level="dataset", observed=True).mean()
        return mean_rank, agreement

    @staticmethod
    def lml_order(runs: pd.DataFrame) -> pd.DataFrame:
        df = Analysis._label(runs)
        lml = df.pivot(index=["dataset", "kernel", "fold"], columns="ard", values="lml")
        below = (lml[True] < lml[False]).groupby(["dataset", "kernel"], observed=True).sum()
        return below.rename("folds with LML(ARD) < LML(iso)").to_frame()

    @staticmethod
    def paired_tests(
            runs: pd.DataFrame,
            pairs: tuple[tuple[str, str], ...] = (
                ("Matérn ARD", "SE ARD"), ("Matérn iso", "SE iso"),
                ("SE ARD", "SE iso"), ("Matérn ARD", "Matérn iso"),
            ),
            metrics: tuple[str, ...] = ("rmse", "nlpd"),
    ) -> pd.DataFrame:
        df = Analysis._scores(runs)
        rows = []
        for name in df.dataset.cat.categories:
            sub = df[df.dataset == name]
            for a, b in pairs:
                A = sub[sub.model == a].set_index("fold")
                B = sub[sub.model == b].set_index("fold")
                if len(A) < 2 or len(A) != len(B):
                    continue
                k, ratio = len(A), (A.n_test / A.n_train).mean()
                for m in metrics:
                    d = (A[m] - B[m]).to_numpy()
                    se = d.std(ddof=1) * np.sqrt(1 / k + ratio)
                    half = student_t.ppf(0.975, k - 1) * se
                    tstat = d.mean() / se
                    rows.append(dict(
                        dataset=name, comparison=f"{a} − {b}", metric=m,
                        mean_diff=d.mean(), ci_low=d.mean() - half, ci_high=d.mean() + half,
                        wins=f"{(d < 0).sum()}/{k}",
                        p_corrected=2 * student_t.sf(abs(tstat), k - 1),
                    ))
        return pd.DataFrame(rows).set_index(["dataset", "comparison", "metric"])

    @staticmethod
    def winners(runs: pd.DataFrame) -> pd.DataFrame:
        df = Analysis._scores(runs)
        mean = df.groupby(["dataset", "model"], observed=True)[list(Analysis.CRITERIA)].mean()
        out = {}
        for c, (label, asc) in Analysis.CRITERIA.items():
            s = mean[c].unstack("model")
            out[label] = s.idxmin(axis=1) if asc else s.idxmax(axis=1)
        out = pd.DataFrame(out)
        out.insert(0, "field", [Analysis.FIELDS[d] for d in out.index])
        return out

    @staticmethod
    @plt.rc_context(STYLE_SUB)
    def plot_relevance(runs: pd.DataFrame) -> None:
        df = Analysis._label(runs)
        df = df[df.ard]
        names = [d for d in df.dataset.cat.categories if d in set(df.dataset)]
        features = {name: Datasets.load(name).features for name in names}
        fig, axes = Analysis._feature_grid(features)
        shown = []
        for ax, name in zip(axes.flat, names):
            sub = df[df.dataset == name]
            models = [m for m in Analysis.MODELS if m in set(sub.model)]
            rel = {
                m: np.exp(-np.log(np.array(
                    [json.loads(s) for s in sub[sub.model == m].length_scale]
                )).mean(axis=0))
                for m in models
            }
            order = np.argsort(rel[models[-1]])
            Analysis._dots(ax, {m: rel[m][order] for m in models}, [features[name][i] for i in order])
            ax.set_xscale("log")
            # Under two decades the decade ticks alone leave the axis almost unlabeled.
            lo, hi = ax.get_xlim()
            if hi / lo < 100:
                ax.xaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            Analysis._title(ax, name, Analysis.FIELDS[name])
            shown += [m for m in models if m not in shown]
        for ax in axes.flat[len(names):]:
            ax.axis("off")
        fig.supxlabel(r"$1/\ell$: nonlinearity along feature (geometric mean over folds)", fontsize=19, color=Analysis.INK2)
        Analysis._legend(fig, shown)
        Analysis._show(fig, "relevance")

    @staticmethod
    def importance(runs: pd.DataFrame, h: float = 1e-3) -> pd.DataFrame:
        df = Analysis._label(runs)
        rows = []
        for name in df.dataset.cat.categories:
            ds = Datasets.load(name)
            folds = Datasets.folds(ds)
            for r in df[df.dataset == name].itertuples():
                tr, te = folds[r.fold]
                sc = StandardScaler().fit(ds.X[tr])
                Xtr, Xte = sc.transform(ds.X[tr]), sc.transform(ds.X[te])
                ls = np.array(json.loads(r.length_scale))
                kernel = GP.kernel(
                    r.kernel, Xtr.shape[1], r.ard, r.signal_var,
                    ls if r.ard else ls[0], r.noise_var,
                )
                gp = GaussianProcessRegressor(kernel, normalize_y=True, optimizer=None).fit(Xtr, ds.y[tr])
                grad = np.empty(Xte.shape[1])
                for j in range(Xte.shape[1]):
                    e = np.zeros(Xte.shape[1])
                    e[j] = h
                    grad[j] = np.mean(np.abs(gp.predict(Xte + e) - gp.predict(Xte - e))) / (2 * h)
                rows.append(dict(
                    dataset=name, model=str(r.model), fold=r.fold,
                    **dict(zip(ds.features, grad / grad.sum())),
                ))
        return pd.DataFrame(rows)

    @staticmethod
    @plt.rc_context(STYLE_SUB)
    def plot_importance(imp: pd.DataFrame) -> None:
        names = [d for d in Config.DATASETS if d in set(imp.dataset)]
        features = {name: Datasets.load(name).features for name in names}
        fig, axes = Analysis._feature_grid(features)
        shown = []
        for ax, name in zip(axes.flat, names):
            feats = features[name]
            mean = imp[imp.dataset == name].groupby("model")[feats].mean()
            models = [m for m in Analysis.MODELS if m in mean.index]
            order = np.argsort(mean.loc[models[-1]].to_numpy())
            Analysis._dots(ax, {m: mean.loc[m].to_numpy()[order] for m in models}, [feats[i] for i in order])
            ax.set_xlim(left=-0.03 * ax.get_xlim()[1])
            Analysis._title(ax, name, Analysis.FIELDS[name])
            shown += [m for m in models if m not in shown]
        for ax in axes.flat[len(names):]:
            ax.axis("off")
        fig.supxlabel("share of mean |∂μ/∂x_j| on test points, mean over folds", fontsize=19, color=Analysis.INK2)
        Analysis._legend(fig, shown)
        Analysis._show(fig, "importance")

    @staticmethod
    def _feature_grid(features: dict[str, list[str]]) -> tuple[plt.Figure, np.ndarray]:
        # Two panels per row; each row is as tall as its longest feature list.
        counts = [len(f) for f in features.values()]
        rows = [max(counts[i:i + 2]) + 2 for i in range(0, len(counts), 2)]
        return plt.subplots(
            len(rows), 2, figsize=(12, 0.42 * sum(rows) + 1.8), squeeze=False,
            height_ratios=rows, layout="constrained",
        )

    @staticmethod
    def _dots(ax: plt.Axes, values: dict[str, np.ndarray], labels: list[str]) -> None:
        # Models are dodged within each feature's row so that equal values stay visible.
        y = np.arange(len(labels))
        offsets = np.linspace(0.5, -0.5, len(values) + 2)[1:-1]
        for row in y[::2]:
            ax.axhspan(row - 0.5, row + 0.5, color=Analysis.BAND, lw=0, zorder=0)
        for dy, (m, v) in zip(offsets, values.items()):
            ax.plot(
                v, y + dy, Analysis.MARKERS[m], ms=10, color=Analysis.COLORS[m],
                mec="white", mew=1, zorder=3, clip_on=False,
            )
        ax.set_yticks(y, [Analysis.SHORT.get(f, f) for f in labels])
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)

    @staticmethod
    def _title(ax: plt.Axes, name: str, field: str) -> None:
        ax.set_title(name)
        ax.set_title(field, loc="right", fontsize=0.9 * plt.rcParams["font.size"], fontweight="normal", color=Analysis.MUTED)

    @staticmethod
    def _legend(fig: plt.Figure, models: list[str], line: bool = False) -> None:
        scale = plt.rcParams["font.size"] / 12
        handles = [
            Line2D(
                [], [], color=Analysis.COLORS[m], lw=2 if line else 0,
                marker=None if line else Analysis.MARKERS[m], ms=7 * scale, mec="white", mew=0.9,
            )
            for m in models
        ]
        fig.legend(handles, models, loc="outside upper center", ncols=len(models), columnspacing=2.2)

    @staticmethod
    def _show(fig: plt.Figure, name: str) -> None:
        figures_dir = Config.RESULTS_DIR / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _label(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ard"] = df["ard"].astype(bool)
        model = df.kernel.map({"rbf": "SE", "matern": "Matérn"}) + df.ard.map({False: " iso", True: " ARD"})
        df["model"] = pd.Categorical(model, Analysis.MODELS, ordered=True)
        order = [d for d in Config.DATASETS if d in set(df.dataset)]
        df["dataset"] = pd.Categorical(df.dataset, order, ordered=True)
        return df

    @staticmethod
    def _scores(runs: pd.DataFrame) -> pd.DataFrame:
        df = Analysis._label(runs)
        df["bic"] = -2 * df.lml + df.n_hyper * np.log(df.n_train)
        df["unexplained"] = 1 - df.r2
        return df

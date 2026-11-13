"""Model harness: GroupKFold CV, boosting, mean baseline, shuffled-label null
and statsmodels MixedLM (player / game / team random intercepts).

Design notes
------------
* Every reported score is *out-of-fold*.  Two groupings are used so nothing
  leaks across folds: GroupKFold by ``nfl_id`` (an athlete appears in exactly
  one fold) and, for play-level targets, GroupKFold by ``game_play_id``.
* Each target is bracketed by (a) a mean dummy regressor fitted on the same
  training folds and (b) a shuffled-label null run through the identical CV --
  a real model must beat both or the score means nothing.
* MixedLM reports two things: the fixed-effect estimates (football meaning:
  which kinematics actually move separation) and a *population-level* OOF score
  (fixed effects only, so unseen players get no fitted random intercept).
* LightGBM is available on this box (4.7.0); scikit-learn's
  HistGradientBoostingRegressor is run alongside it as an independent check.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

try:  # pragma: no cover - environment dependent
    from lightgbm import LGBMRegressor
    HAVE_LIGHTGBM = True
except Exception:  # pragma: no cover
    LGBMRegressor = None
    HAVE_LIGHTGBM = False

RANDOM_STATE = 42
N_SPLITS = 5


# --------------------------------------------------------------- estimation --
def make_regressor(kind: str = "lightgbm", n_rows: int = 10_000):
    """Return an unfitted regressor: lightgbm, hist_gb or mean dummy."""
    if kind == "mean":
        return _MeanRegressor()
    if kind == "hist_gb":
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20 if n_rows > 2000 else 5,
            l2_regularization=1.0, random_state=RANDOM_STATE)
    if kind == "lightgbm":
        if not HAVE_LIGHTGBM:
            raise RuntimeError("lightgbm not importable; use hist_gb")
        small = n_rows < 2000
        return LGBMRegressor(
            n_estimators=400, learning_rate=0.05,
            num_leaves=15 if small else 31,
            min_child_samples=3 if small else 20,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=4)
    raise ValueError(f"unknown model kind {kind!r}")


class _MeanRegressor:
    """Training-fold mean predictor (the baseline every score is read against)."""

    def fit(self, X, y):
        self.m_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m_)


# ----------------------------------------------------------------- features --
def prepare_matrix(df: pd.DataFrame, feature_cols: Sequence[str],
                   drop: Sequence[str] = ()) -> Tuple[pd.DataFrame, List[str]]:
    """Select + one-hot the requested features (ids and targets never enter)."""
    cols = [c for c in feature_cols if c in df.columns and c not in drop]
    if not cols:
        raise ValueError("no usable feature columns")
    X = df[cols].copy()
    obj = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "category"]
    if obj:
        X = pd.get_dummies(X, columns=obj, dummy_na=False, dtype=float)
    return X, cols


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else np.nan
    return dict(r2=float(r2_score(y_true, y_pred)),
                mae=float(mean_absolute_error(y_true, y_pred)),
                rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                spearman=float(rho) if rho == rho else float("nan"))


def run_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, kind: str,
           n_splits: int = N_SPLITS, shuffle_labels: bool = False,
           seed: int = RANDOM_STATE) -> Dict:
    """GroupKFold CV with train-fold median imputation and OOF collection."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    if shuffle_labels:
        y = rng.permutation(y)
    n_splits = int(min(n_splits, len(np.unique(groups))))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    fold_r2, fold_mae = [], []
    imp = None
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med).fillna(0.0)
        Xte = Xte.fillna(med).fillna(0.0)
        model = make_regressor(kind, n_rows=len(tr))
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        oof[te] = pred
        fold_r2.append(float(r2_score(y[te], pred)))
        fold_mae.append(float(mean_absolute_error(y[te], pred)))
        if hasattr(model, "feature_importances_"):
            imp_ = np.asarray(model.feature_importances_, dtype=float)
            imp = imp_ if imp is None else imp + imp_
    res = _score(y[np.isfinite(oof)], oof[np.isfinite(oof)])
    res.update(kind=kind, n_splits=n_splits, fold_r2=fold_r2, fold_mae=fold_mae,
               oof=oof, y_true=y, shuffled_labels=bool(shuffle_labels))
    if imp is not None:
        gain = pd.Series(imp / n_splits, index=X.columns)
        res["importance"] = {k: float(v) for k, v in
                             gain.sort_values(ascending=False).head(25).items()}
    return res


def evaluate_target(df: pd.DataFrame, target: str, feature_cols: Sequence[str],
                    groupings: Dict[str, np.ndarray],
                    kinds: Sequence[str] = ("lightgbm", "hist_gb"),
                    n_splits: int = N_SPLITS) -> Dict:
    """Full evaluation of one target: models + mean baseline + shuffled null.

    ``groupings`` maps a split name (e.g. ``by_player``) to a group vector
    aligned with ``df``'s rows, so the same target can be scored with players
    held out *and* with games held out.
    """
    mask = df[target].notna().to_numpy()
    d = df[mask].copy()
    y = d[target].to_numpy(float)
    X, used = prepare_matrix(d, feature_cols, drop=[target])
    out = dict(target=target, rows=int(len(d)),
               players=int(d["nfl_id"].nunique()) if "nfl_id" in d.columns else None,
               n_features=int(X.shape[1]), feature_columns=used,
               groupings={})
    for gname, groups in groupings.items():
        g = np.asarray(groups)[mask]
        res = {"models": {}, "baselines": {}}
        for kind in kinds:
            try:
                res["models"][kind] = run_cv(X, y, g, kind, n_splits)
            except Exception as exc:  # pragma: no cover - environment dependent
                res["models"][kind] = {"error": str(exc)}
        res["baselines"]["mean_dummy"] = run_cv(X, y, g, "mean", n_splits)
        res["baselines"]["shuffled_label_null"] = run_cv(
            X, y, g, kinds[0], n_splits, shuffle_labels=True)
        out["groupings"][gname] = res
    return out


def strip_heavy(res: Dict) -> Dict:
    """Drop OOF arrays / fold internals so a result dict is JSON-serialisable."""
    out = {}
    for k, v in res.items():
        if k in ("oof", "y_true", "fold_r2", "fold_mae"):
            out[k] = None if k in ("oof", "y_true") else \
                [round(float(x), 4) for x in v]
        elif isinstance(v, dict):
            out[k] = strip_heavy(v)
        elif isinstance(v, list):
            out[k] = v
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, float):
            out[k] = round(v, 6)
        else:
            out[k] = v
    return out


# ------------------------------------------------------------ mixed effects --
def _fit_mixedlm(formula: str, d: pd.DataFrame, group_col: str,
                 vc: Optional[Dict[str, str]] = None, reml: bool = True,
                 y: Optional[np.ndarray] = None):
    """Fit one MixedLM, trying lbfgs -> bfgs -> powell.

    If ``y`` is given, optimiser output that hits the degenerate boundary
    solution (Intercept = 0 while ``mean(y) != 0``) is skipped in favour of the
    next optimiser: ``lbfgs`` can report ``converged=True`` with ``llf = inf``,
    which is not a fit at all.
    """
    import statsmodels.formula.api as smf
    kwargs = {"vc_formula": vc} if vc else {}
    last: Optional[Exception] = None
    first = None
    for meth in ("lbfgs", "bfgs", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(formula, d, groups=d[group_col], **kwargs).fit(
                    reml=reml, method=meth, maxiter=1000, disp=False)
        except Exception as exc:          # try the next optimiser
            last = exc
            continue
        if y is None or not _degenerate(fit, y):
            return fit
        if first is None:
            first = fit                  # remember it in case all are degenerate
    if first is not None:
        return first
    raise last if last is not None else RuntimeError("MixedLM failed")


def _degenerate(mdf, y: np.ndarray) -> bool:
    """Boundary solution: REML collapsed the intercept while y has a mean.

    statsmodels returns `converged=True` for this solution, but its fixed
    effects are useless (Intercept = 0 while the target mean is ~3 yd), which
    would show up as a nonsensical in-sample R2.  Such fits are refit by ML.
    """
    try:
        ip = float(mdf.fe_params.get("Intercept", 0.0))
    except Exception:
        return False
    return bool(abs(ip) < 1e-6 and abs(float(np.mean(y))) > 0.05)


def _fit_fixed_effects(mdf):
    """X @ beta on the model's own design matrix (never raises, always aligned)."""
    return np.asarray(mdf.model.exog, dtype=float) @ \
        np.asarray(mdf.fe_params, dtype=float)

def _formula_vars(formula: str, columns: Sequence[str]) -> List[str]:
    """Formula-referenced names that actually exist in ``columns``."""
    names = _vars(formula)
    return [n for n in names if n in columns]


def run_mixedlm(df: pd.DataFrame, formula: str, group_col: str,
                vc: Optional[Dict[str, str]] = None, label: str = "") -> Dict:
    """One statsmodels MixedLM fit with a random intercept per ``group_col``.

    ``vc`` optionally adds a second variance component (e.g.
    ``{'team': '0 + C(team_abbr)'}`` or ``{'game': '0 + C(game_id)'}``) so
    player *and* game/team effects are estimated at once.  Convergence,
    coefficients and p-values are returned verbatim.
    """
    import statsmodels.formula.api as smf  # noqa: F401  (kept for callers)
    d = df.copy()
    target = formula.split("~")[0].strip()
    # NOTE: the target must be in the dropna set too - patsy drops rows with a
    # missing endog, and a groups vector that is longer than the design matrix
    # makes statsmodels raise "index N is out of bounds" in group_list().
    used = _formula_vars(formula, d.columns) + [group_col, target]
    used = list(dict.fromkeys(used))
    for c in used:
        if c in d.columns and d[c].dtype == object:
            d[c] = d[c].astype("category")
    # drop incomplete rows up-front: patsy would otherwise shrink the frame
    # while `groups` keeps its original length
    d = d.dropna(subset=[c for c in used if c in d.columns]).copy()
    if d[group_col].nunique() < 3 or len(d) < 50:
        return dict(label=label, skipped="too few rows/groups")
    yv = d[target].to_numpy(float)
    try:
        mdf = _fit_mixedlm(formula, d, group_col, vc=vc, reml=True, y=yv)
    except Exception as exc:
        return dict(label=label, converged=False, error=str(exc)[:200])
    reml_used, ml_refit, note = "REML", False, None
    if _degenerate(mdf, yv):
        # REML hit a boundary solution (Intercept = 0 while the target mean is
        # not 0): coefficients are meaningless, so refit by maximum likelihood
        try:
            alt = _fit_mixedlm(formula, d, group_col, vc=vc, reml=False, y=yv)
            if not _degenerate(alt, yv):
                mdf, reml_used, ml_refit = alt, "ML", True
                note = ("REML collapsed to a boundary solution (Intercept = 0, "
                        "zero random-effect variance); refit by ML")
        except Exception as exc:
            note = f"ML refit failed: {str(exc)[:140]}"
    singular = False
    try:
        re_sd = float(np.sqrt(float(mdf.cov_re.iloc[0, 0]))) \
            if mdf.cov_re is not None and mdf.cov_re.size else float("nan")
    except Exception:
        re_sd = float("nan")
    fe_fitted = _fit_fixed_effects(mdf)
    try:
        fitted = np.asarray(mdf.fittedvalues, dtype=float)
    except Exception as exc:
        # singular random-effects covariance: fittedvalues/BLUPs undefined,
        # so the in-sample score uses the fixed effects only (flagged below)
        singular = True
        note = (note + " | " if note else "") + str(exc)[:140]
        fitted = fe_fitted
    return dict(
        label=label, converged=bool(mdf.converged), group_col=group_col,
        vc=vc, rows=int(len(d)), groups=int(d[group_col].nunique()),
        reml=reml_used, ml_refit=bool(ml_refit),
        singular_random_effects=singular, note=note,
        r2_fitted_insample=float(r2_score(yv, fitted)),
        r2_fixed_effects_insample=float(r2_score(yv, fe_fitted)),
        loglike=float(mdf.llf),
        params={k: float(v) for k, v in mdf.params.items()},
        pvalues={k: float(v) for k, v in mdf.pvalues.items()},
        random_intercept_sd=re_sd,
    )


def run_mixedlm_cv(df: pd.DataFrame, formula: str, feature_cols: Sequence[str],
                   group_col: str = "nfl_id", n_splits: int = N_SPLITS,
                   vc: Optional[Dict[str, str]] = None,
                   label: str = "") -> Dict:
    """Population-level OOF score for MixedLM under GroupKFold by player.

    Test folds are predicted from the *fixed effects only*: a held-out player
    has no estimated random intercept, exactly as in the boosting runs, so the
    numbers are comparable.
    """
    import statsmodels.formula.api as smf
    target = formula.split("~")[0].strip()
    used = _formula_vars(formula, df.columns) + [group_col, target]
    d = df.copy()
    for c in used:
        if c in d.columns and d[c].dtype == object:
            d[c] = d[c].astype("category")
    d = d.dropna(subset=[c for c in used if c in d.columns]).copy()
    y = d[target].to_numpy(float)
    groups = d[group_col].to_numpy()
    n_splits = int(min(n_splits, len(np.unique(groups))))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    converged, fallback, ml_refits, errors = 0, 0, 0, []
    for tr, te in gkf.split(d, y, groups):
        dtr, dte = d.iloc[tr], d.iloc[te]
        try:
            ytr = dtr[target].to_numpy(float)
            mdf = _fit_mixedlm(formula, dtr, group_col, vc=vc, reml=True, y=ytr)
            if _degenerate(mdf, ytr):
                # same REML boundary problem as the full fit: use ML so the
                # held-out fixed-effect predictions keep their intercept
                alt = _fit_mixedlm(formula, dtr, group_col, vc=vc, reml=False,
                                   y=ytr)
                if not _degenerate(alt, ytr):
                    mdf = alt
                    ml_refits += 1
            fe = [c for c in mdf.fe_params.index]
            # rebuild the training fixed-effect design for the test frame
            Xt = None
            di = getattr(getattr(mdf, "model", None), "data", None)
            di = getattr(di, "design_info", None)
            if di is not None:
                try:
                    import patsy
                    Xt = np.asarray(patsy.build_design_matrices([di], dte)[0],
                                    dtype=float)
                except Exception:
                    Xt = None
            if Xt is None:
                Xt = _design(mdf, formula, dte).astype(float)
            if Xt.shape[1] != len(fe):          # pragma: no cover - patsy drift
                raise ValueError(f"design mismatch {Xt.shape} vs {len(fe)}")
            pred = Xt @ mdf.fe_params.to_numpy(float)
            if not np.all(np.isfinite(pred)):
                raise ValueError("non-finite MixedLM fixed-effect prediction")
            oof[te] = pred
            converged += int(bool(mdf.converged))
            if not bool(mdf.converged):
                fallback += 1
        except Exception as exc:
            # Honest population-only fallback: OLS with identical fixed
            # effects, fit on the training fold only (never sees the test fold).
            errors.append(str(exc)[:160])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ols = smf.ols(formula, dtr).fit()
                    oof[te] = np.asarray(ols.predict(dte), dtype=float)
                fallback += 1
            except Exception as exc2:            # pragma: no cover - degenerate fold
                errors.append("ols:" + str(exc2)[:120])
    m = np.isfinite(oof)
    res = _score(y[m], oof[m]) if m.sum() > 2 else dict(r2=np.nan, mae=np.nan,
                                                        rmse=np.nan, spearman=np.nan)
    res.update(label=label, model="mixedlm_population", group_col=group_col,
               n_splits=n_splits, folds_converged=converged,
               folds_ml_refit=ml_refits,
               folds_fallback=fallback, errors=errors,
               n_rows=int(m.sum()), oof=oof)
    return res


def _design(mdf, formula: str, d: pd.DataFrame) -> np.ndarray:
    """Rebuild the fixed-effect design matrix of ``mdf`` for a new frame."""
    import patsy
    _, Xt = patsy.dmatrices(formula, d, return_type="dataframe")
    names = list(mdf.fe_params.index)
    Xt = Xt.reindex(columns=names, fill_value=0.0)
    return Xt.to_numpy(float)


def _vars(formula: str) -> List[str]:
    """Column names referenced by a patsy formula (best effort)."""
    import re
    return list(dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula.split("~")[-1])))

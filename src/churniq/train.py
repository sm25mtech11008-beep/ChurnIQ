"""Phase B training entrypoint.

    python -m churniq.train [--trials 20] [--cv-rows N] [--quick]

Steps:
 1. Stratified 5-fold CV: logistic baseline vs default LightGBM (model selection
    with honest error bars — a single split can flatter either model).
 2. Optuna hyperparameter search on a stratified subsample (search on a
    subsample, refit on everything: tuning ranks configurations, and ranks are
    stable under subsampling; final quality comes from the full-data refit).
 3. Final fit on 80% train, isotonic calibration on a held-out calibration
    slice (never the test set), evaluation on the untouched 20% holdout.
 4. Business threshold from the expected-savings curve on the holdout.
 5. Save model bundle + metrics.json + model_card.md + figures.
"""

import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config
from .data import load_clean, split_xy
from .evaluate import best_threshold, classification_metrics, decile_lift, savings_curve
from .pipeline import build_lgbm, build_logistic


def stratified_subsample(X, y, n, seed=config.RANDOM_STATE):
    if n is None or n >= len(y):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys


def cv_compare(X, y, cv_rows, n_splits=5):
    """Cross-validated ROC-AUC for baseline vs LightGBM."""
    Xc, yc = stratified_subsample(X, y, cv_rows)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    results = {}
    for name, model in [("logistic_baseline", build_logistic()), ("lightgbm_default", build_lgbm())]:
        t0 = time.time()
        scores = cross_val_score(model, Xc, yc, cv=skf, scoring="roc_auc", n_jobs=1)
        results[name] = {
            "cv_auc_mean": float(np.mean(scores)),
            "cv_auc_std": float(np.std(scores)),
            "cv_rows": len(yc),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  {name}: AUC {np.mean(scores):.4f} +/- {np.std(scores):.4f} ({results[name]['seconds']}s)")
    return results


def tune(X, y, n_trials, sample_rows=200_000):
    """Optuna search (TPE) maximizing 3-fold ROC-AUC on a stratified subsample."""
    Xs, ys = stratified_subsample(X, y, sample_rows)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 500),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
        return cross_val_score(build_lgbm(**params), Xs, ys, cv=skf, scoring="roc_auc", n_jobs=1).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  best trial AUC {study.best_value:.4f} with {study.best_params}")
    return study.best_params, float(study.best_value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="parquet path (default: clean_full.parquet)")
    ap.add_argument("--trials", type=int, default=20, help="optuna trials (0 = skip tuning)")
    ap.add_argument("--cv-rows", type=int, default=None, help="subsample rows for CV compare")
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke-testing")
    args = ap.parse_args()

    if args.quick:
        args.cv_rows = args.cv_rows or 20_000
        args.trials = min(args.trials, 5)

    t_start = time.time()
    print("Loading data ...")
    df = load_clean(args.data)
    X, y = split_xy(df)
    print(f"  {len(y):,} rows, churn rate {y.mean():.4f}")

    print("Step 1/4: 5-fold CV — logistic baseline vs LightGBM")
    cv_results = cv_compare(X, y, args.cv_rows)

    best_params, tuned_cv_auc = {}, None
    if args.trials > 0:
        print(f"Step 2/4: Optuna tuning ({args.trials} trials on 200k subsample)")
        best_params, tuned_cv_auc = tune(X, y, args.trials)

    print("Step 3/4: final fit + isotonic calibration + holdout evaluation")
    # 80/20 holdout; within the 80%, hold out 15% purely for calibration.
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=config.RANDOM_STATE
    )
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=config.RANDOM_STATE
    )
    raw = build_lgbm(**best_params)
    raw.fit(X_fit, y_fit)

    calibrated = CalibratedClassifierCV(FrozenEstimator(raw), method="isotonic")
    calibrated.fit(X_cal, y_cal)

    p_raw = raw.predict_proba(X_hold)[:, 1]
    p_cal = calibrated.predict_proba(X_hold)[:, 1]
    metrics_raw = classification_metrics(y_hold, p_raw)
    metrics_cal = classification_metrics(y_hold, p_cal)
    print(f"  raw       : {metrics_raw}")
    print(f"  calibrated: {metrics_cal}")

    print("Step 4/4: business threshold from expected-savings curve")
    curve = savings_curve(y_hold.to_numpy(), p_cal, X_hold["monthlycharges"].to_numpy())
    chosen = best_threshold(curve)
    print(f"  chosen: {chosen}")
    lift_table = decile_lift(y_hold, p_cal)

    # ---- figures -----------------------------------------------------------
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for probs, label in [(p_raw, "raw LightGBM"), (p_cal, "isotonic-calibrated")]:
        frac_pos, mean_pred = calibration_curve(y_hold, probs, n_bins=15, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 0.5], [0, 0.5], "k--", lw=1, label="perfect")
    ax.set(xlabel="mean predicted probability", ylabel="observed churn rate", title="Calibration (holdout)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "calibration_curve.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["threshold"], curve["expected_savings"] / 1000, lw=2)
    ax.axvline(chosen["threshold"], color="crimson", ls="--", label=f"chosen t={chosen['threshold']:.2f}")
    ax.set(xlabel="targeting threshold on P(churn)", ylabel="expected savings ($k)",
           title="Retention-campaign expected savings (holdout)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "savings_curve.png", dpi=120)

    booster = raw.named_steps["clf"].booster_
    imp = sorted(
        zip(raw.named_steps["prep"].get_feature_names_out(), booster.feature_importance("gain")),
        key=lambda kv: kv[1], reverse=True,
    )[:15]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([n.split("__", 1)[-1] for n, _ in imp][::-1], [v for _, v in imp][::-1])
    ax.set(title="LightGBM feature importance (gain), top 15", xlabel="total gain")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close("all")

    # ---- artifacts ---------------------------------------------------------
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(y)),
        "churn_rate": float(y.mean()),
        "cv_compare": cv_results,
        "optuna_trials": args.trials,
        "tuned_cv_auc": tuned_cv_auc,
        "best_params": best_params,
        "holdout_metrics_raw": metrics_raw,
        "holdout_metrics_calibrated": metrics_cal,
        "business_threshold": chosen,
        "business_assumptions": {
            "offer_cost": config.OFFER_COST,
            "save_rate": config.SAVE_RATE,
            "clv_months": config.CLV_MONTHS,
        },
        "top_gain_features": [n.split("__", 1)[-1] for n, _ in imp[:8]],
        "total_seconds": round(time.time() - t_start, 1),
    }
    joblib.dump(
        {"model": calibrated, "raw": raw, "threshold": chosen["threshold"],
         "feature_cols": config.FEATURE_COLS, "metadata": metadata},
        config.MODEL_PATH,
    )
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2))
    lift_table.to_csv(config.REPORTS_DIR / "decile_lift.csv", index=False)
    _write_model_card(metadata, lift_table)
    print(f"Done in {metadata['total_seconds']}s -> {config.MODEL_PATH}")


def _write_model_card(m: dict, lift_table) -> None:
    cal = m["holdout_metrics_calibrated"]
    thr = m["business_threshold"]
    top_decile = lift_table.iloc[0]
    card = f"""# ChurnIQ Model Card

**Model:** LightGBM (isotonic-calibrated) inside a leakage-free sklearn pipeline.
**Trained:** {m["trained_at"]} on {m["n_rows"]:,} rows (churn rate {m["churn_rate"]:.2%}).

## Selection

5-fold stratified CV, ROC-AUC:

| model | CV AUC |
|---|---|
| Logistic baseline | {m["cv_compare"]["logistic_baseline"]["cv_auc_mean"]:.4f} ± {m["cv_compare"]["logistic_baseline"]["cv_auc_std"]:.4f} |
| LightGBM (default) | {m["cv_compare"]["lightgbm_default"]["cv_auc_mean"]:.4f} ± {m["cv_compare"]["lightgbm_default"]["cv_auc_std"]:.4f} |

Optuna ({m["optuna_trials"]} TPE trials, 3-fold on 200k subsample) best CV AUC: {m["tuned_cv_auc"] if m["tuned_cv_auc"] else "n/a"}.

## Holdout performance (20%, untouched during training/tuning/calibration)

| metric | value | context |
|---|---|---|
| ROC-AUC | {cal["roc_auc"]:.4f} | 0.5 = random |
| PR-AUC | {cal["pr_auc"]:.4f} | base rate {m["churn_rate"]:.3f} = random |
| Brier | {cal["brier"]:.4f} | lower is better; churn-rate-only predictor ≈ {m["churn_rate"] * (1 - m["churn_rate"]):.4f} |

Phase A predicted a weak-signal dataset (best single-feature AUC 0.576) — results should be read
against that ceiling, not against tutorial datasets.

## Business threshold

Chosen to maximize expected retention-campaign savings on the holdout
(offer cost ${m["business_assumptions"]["offer_cost"]:.0f}, save rate {m["business_assumptions"]["save_rate"]:.0%},
CLV = {m["business_assumptions"]["clv_months"]} × monthly charge):

- **threshold = {thr["threshold"]:.2f}** → target {thr["n_targeted"]:,} customers,
  catch {thr["true_churners_caught"]:,} true churners,
  expected savings **${thr["expected_savings"]:,.0f}** on the holdout slice.

## Sanity check vs EDA

Top gain features: {", ".join(f"`{f}`" for f in m["top_gain_features"])}.
Phase A's top separators (customer_satisfaction, num_complaints, num_service_calls,
late_payments, contract) should dominate this list — if they don't, investigate before trusting the model.

Top-decile lift: predicted-risk decile 10 churns at {top_decile["churn_rate"]:.2%}
({top_decile["lift"]:.2f}× base rate).

## Known limitations

- Synthetic data; no true temporal split (no event timestamps beyond signup date).
- Business assumptions (offer cost / save rate / CLV) are placeholders to demonstrate
  the method — sensitivity analysis belongs in front of any real deployment.
"""
    (config.REPORTS_DIR / "model_card.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()

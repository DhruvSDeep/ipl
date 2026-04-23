"""Entry point for the IPL 1st-innings score prediction pipeline.

Run this file to execute the full two-stage pipeline. All model logic
lives in `modelLogic.py` and is called from here.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelLogic import (
    ARCHIVE,
    TARGET,
    TEST_SEASONS,
    VAL_FRAC,
    PlayerVocab,
    ScalarVocab,
    build_features,
    build_rolling_form,
    evaluate,
    evaluate_by_season,
    evaluate_ensemble,
    evaluate_ensemble_by_season,
    fit_season_baseline,
    load_data,
    load_player_stats,
    season_mean_baseline,
    time_split_mask,
    train_ensemble,
    train_model,
    train_player_encoder,
    train_val_split_mask,
    trend_baseline_metrics,
)


def _banner(text: str, char: str = "=") -> None:
    line = char * (len(text) + 6)
    print(f"\n{line}\n== {text} ==\n{line}")


def _print_metrics_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    name_w = max(len(n) for n, _ in rows)
    header = f"  {'split'.ljust(name_w)}   {'MAE':>7}   {'RMSE':>7}   {'n':>5}"
    print(header)
    print(f"  {'-' * name_w}   {'-' * 7}   {'-' * 7}   {'-' * 5}")
    for name, m in rows:
        n_str = f"{m['n']:5d}" if "n" in m else "     "
        print(f"  {name.ljust(name_w)}   {m['MAE']:7.2f}   {m['RMSE']:7.2f}   {n_str}")


def show_dashboard(
    stage1_history: dict,
    pred_te: np.ndarray,
    y_te: pd.Series,
    seasons_te: pd.Series,
    overall_metrics: dict,
    season_metrics: dict,
    baseline_season_metrics: dict,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("IPL 1st-innings Score Prediction", fontsize=14, fontweight="bold")

    # Top-left: Stage 1 loss curve
    ax = axes[0, 0]
    train_hist = stage1_history["train"]
    val_hist = stage1_history["val"]
    best_ep = stage1_history["best_epoch"]
    ax.plot(train_hist, color="#2a6ebb", linewidth=1.5, label="train")
    ax.plot(val_hist, color="#d1495b", linewidth=1.5, label="val")
    ax.axvline(best_ep, linestyle="--", color="gray", alpha=0.6,
               label=f"best val @ ep {best_ep + 1}")
    ax.set_title("Stage 1 · PlayerEncoder Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (runs)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: predicted vs actual scatter, coloured by season
    ax = axes[0, 1]
    y_arr = y_te.to_numpy()
    unique_seasons = sorted(seasons_te.unique())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(unique_seasons)))
    for season, color in zip(unique_seasons, colors):
        mask = (seasons_te == season).to_numpy()
        ax.scatter(y_arr[mask], pred_te[mask], alpha=0.65, s=30,
                   color=color, label=str(season))
    lo = min(y_arr.min(), pred_te.min()) - 5
    hi = max(y_arr.max(), pred_te.max()) + 5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.35, linewidth=1, label="perfect")
    ax.set_title("Predicted vs Actual (test)")
    ax.set_xlabel("actual runs")
    ax.set_ylabel("predicted runs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: per-season MAE bar chart
    ax = axes[1, 0]
    seasons_sorted = sorted(season_metrics)
    x = np.arange(len(seasons_sorted))
    w = 0.35
    baseline_maes = [baseline_season_metrics.get(s, {}).get("MAE", 0) for s in seasons_sorted]
    lgbm_maes = [season_metrics[s]["MAE"] for s in seasons_sorted]
    ns = [season_metrics[s]["n"] for s in seasons_sorted]
    bars1 = ax.bar(x - w / 2, baseline_maes, w, label="baseline", color="#aec6cf")
    bars2 = ax.bar(x + w / 2, lgbm_maes, w, label="LGBM", color="#2a6ebb")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(seasons_sorted, ns)])
    ax.set_title("Test MAE by Season")
    ax.set_ylabel("MAE (runs)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # Bottom-right: summary metrics table
    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ["baseline (test)", f"{overall_metrics['baseline']['MAE']:.2f}",
         f"{overall_metrics['baseline']['RMSE']:.2f}", ""],
        ["LGBM (val)", f"{overall_metrics['val']['MAE']:.2f}",
         f"{overall_metrics['val']['RMSE']:.2f}", ""],
        ["LGBM (test)", f"{overall_metrics['test']['MAE']:.2f}",
         f"{overall_metrics['test']['RMSE']:.2f}", ""],
    ]
    for s in seasons_sorted:
        m = season_metrics[s]
        rows.append([f"LGBM ({s})", f"{m['MAE']:.2f}", f"{m['RMSE']:.2f}", str(m['n'])])
    table = ax.table(
        cellText=rows,
        colLabels=["Split", "MAE", "RMSE", "n"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.7)
    ax.set_title("Summary Metrics", pad=12)

    fig.tight_layout()
    plt.show()


def main():
    _banner("IPL 1st-innings score prediction · pipeline start")
    t_total = time.perf_counter()

    _banner("Phase 0 · Data & vocabularies", char="-")
    df = load_data()
    is_train_full = time_split_mask(df)
    is_train, is_val = train_val_split_mask(df, is_train_full)
    is_test = ~is_train_full

    player_vocab = PlayerVocab.from_df(df)
    venue_vocab = ScalarVocab.from_series(df[is_train]["venue"])
    raw_stats = load_player_stats(player_vocab)
    missing = (raw_stats[1:].sum(axis=1) == 0).sum()
    n_matches = df["match_id"].nunique()
    print(f"  rows         : {len(df):,}  ({n_matches:,} matches × 2 innings)")
    print(f"  player vocab : {len(player_vocab) - 1:,} ({missing} without career stats)")
    print(f"  venue vocab  : {len(venue_vocab) - 1}")
    print(f"  test seasons : {', '.join(TEST_SEASONS)}")
    print(f"  splits       : train={is_train.sum()}  "
          f"val={is_val.sum()}  test={is_test.sum()}  "
          f"(val ≈ {VAL_FRAC:.0%} of train matches, split by match_id)")

    _banner("Phase 1 · PlayerEncoder (PyTorch, multi-task)", char="-")
    t1 = time.perf_counter()
    player_emb, venue_emb, stage1_history = train_player_encoder(
        df, player_vocab, venue_vocab,
        train_mask=is_train, val_mask=is_val,
    )
    print(f"  phase 1 elapsed: {time.perf_counter() - t1:.1f}s")

    emb_path = ARCHIVE / "player_embeddings.npz"
    names_arr = np.empty(len(player_vocab), dtype=object)
    for name, idx in player_vocab.id_of.items():
        names_arr[idx] = name
    np.savez(emb_path, embeddings=player_emb, names=names_arr)
    print(f"  saved embeddings → {emb_path}")

    _banner("Phase 2 · LightGBM on pooled features (innings==1 only)", char="-")
    t2 = time.perf_counter()
    # Stage 2 predicts the 1st-innings team total — restrict to inn1 rows.
    inn1 = (df["innings"] == 1).to_numpy()
    df_inn1 = df[inn1].reset_index(drop=True)
    is_train_inn1 = is_train.to_numpy()[inn1]
    is_val_inn1 = is_val.to_numpy()[inn1]
    is_test_inn1 = is_test.to_numpy()[inn1]
    is_train_full_inn1 = is_train_full.to_numpy()[inn1]

    rolling_form = build_rolling_form(df_inn1)
    ds = build_features(
        df_inn1, player_vocab, player_emb, venue_vocab, venue_emb, raw_stats,
        rolling_form=rolling_form,
    )
    print(f"  feature matrix: {ds.X.shape[0]:,} rows × {ds.X.shape[1]} cols")

    # Season baseline: per-year mean for observed train years, weighted-
    # recency extrapolation for unseen future years. LGBM models the
    # residual around this level so predictions for 2024/2025 aren't
    # capped at the maximum training-leaf value. The weighted-recency
    # extrapolation corrects the naive linear fit which severely
    # undershoots the post-2022 scoring jump.
    trend = fit_season_baseline(df_inn1[is_train_inn1])
    trend_all = trend.predict(df_inn1["year"].to_numpy())
    print(f"  season base  : slope={trend.slope:+.2f} runs/year  "
          f"intercept={trend.intercept:.1f}  "
          f"(τ-weighted; max_train_year={trend.max_train_year})  "
          f"→ 2023 base={trend.predict([2023])[0]:.1f}  "
          f"2024 base={trend.predict([2024])[0]:.1f}  "
          f"2025 base={trend.predict([2025])[0]:.1f}")

    y_detr = pd.Series(ds.y.to_numpy() - trend_all, index=ds.y.index)
    X_tr, y_tr = ds.X[is_train_inn1], y_detr[is_train_inn1]
    X_val, y_val = ds.X[is_val_inn1], y_detr[is_val_inn1]
    X_te = ds.X[is_test_inn1]
    y_te = ds.y[is_test_inn1].reset_index(drop=True)
    y_val_raw = ds.y[is_val_inn1]
    seasons_te = ds.seasons[is_test_inn1].reset_index(drop=True)
    offset_val = trend_all[is_val_inn1]
    offset_te = trend_all[is_test_inn1]

    model = train_ensemble(
        X_tr, y_tr, X_val, y_val,
        categorical_cols=ds.categorical_cols,
        n_models=5,
    )
    print(f"  phase 2 elapsed: {time.perf_counter() - t2:.1f}s  "
          f"(ensemble of {len(model.boosters)} boosters)")

    _banner("Results", char="-")
    baseline = season_mean_baseline(df_inn1[is_train_full_inn1], df_inn1[is_test_inn1])
    trend_baseline = trend_baseline_metrics(trend, df_inn1[is_test_inn1])
    val_metrics = evaluate_ensemble(model, X_val, y_val_raw, offset=offset_val)
    test_metrics = evaluate_ensemble(model, X_te, y_te, offset=offset_te)
    season_metrics = evaluate_ensemble_by_season(
        model, X_te, y_te, seasons_te, offset=offset_te
    )

    # Overall baseline per season (both test seasons not in train → predict overall mean)
    overall_mean = df_inn1[is_train_full_inn1][TARGET].mean()
    baseline_by_season: dict[str, dict] = {}
    for season, grp in df_inn1[is_test_inn1].groupby("season", observed=True):
        pred_base = pd.Series(overall_mean, index=grp.index)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        baseline_by_season[str(season)] = {
            "MAE": float(mean_absolute_error(grp[TARGET], pred_base)),
            "RMSE": float(np.sqrt(mean_squared_error(grp[TARGET], pred_base))),
            "n": len(grp),
        }

    _print_metrics_table([
        ("baseline (test)", baseline),
        ("LGBM  (val)   ", val_metrics),
        ("LGBM  (test)  ", test_metrics),
        *[(f"  LGBM  ({s})  ", m) for s, m in season_metrics.items()],
    ])

    print(f"\ntotal pipeline: {time.perf_counter() - t_total:.1f}s")

    # Final predictions include the season-baseline offset so diagnostics
    # reflect what the user actually sees (not the detrended residual).
    pred_te = model.predict(X_te) + offset_te

    _banner("Diagnostic · prediction spread & feature importance", char="-")
    y_te_arr = y_te.to_numpy()
    print(f"  y    : mean={y_te_arr.mean():6.1f}  std={y_te_arr.std():5.1f}  "
          f"range=[{y_te_arr.min():.0f}, {y_te_arr.max():.0f}]")
    print(f"  pred : mean={pred_te.mean():6.1f}  std={pred_te.std():5.1f}  "
          f"range=[{pred_te.min():.1f}, {pred_te.max():.1f}]")
    print(f"  spread ratio (pred.std / y.std): {pred_te.std() / y_te_arr.std():.3f}")
    # Average feature importance across the ensemble for a stable ranking.
    imp_matrix = np.stack([
        b.feature_importance(importance_type="gain") for b in model.boosters
    ])
    imp = pd.Series(
        imp_matrix.mean(axis=0),
        index=X_tr.columns,
    ).sort_values(ascending=False)
    print(f"\n  top 15 features by gain:")
    for name, val in imp.head(15).items():
        print(f"    {name:30s}  {val:10.1f}")
    print(f"\n  zero-gain features: {(imp == 0).sum()} / {len(imp)}")

    show_dashboard(
        stage1_history=stage1_history,
        pred_te=pred_te,
        y_te=y_te,
        seasons_te=seasons_te,
        overall_metrics={"baseline": baseline, "val": val_metrics, "test": test_metrics},
        season_metrics=season_metrics,
        baseline_season_metrics=baseline_by_season,
    )

    return model, player_emb, venue_emb


if __name__ == "__main__":
    main()

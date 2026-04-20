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
    TARGET,
    TEST_SEASONS,
    VAL_FRAC,
    PlayerVocab,
    ScalarVocab,
    build_features,
    evaluate,
    evaluate_by_season,
    load_data,
    load_player_stats,
    season_mean_baseline,
    time_split_mask,
    train_model,
    train_player_encoder,
    train_val_split_mask,
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
    raw_stats, std_stats = load_player_stats(player_vocab)
    missing = (raw_stats[1:].sum(axis=1) == 0).sum()
    print(f"  rows         : {len(df):,}")
    print(f"  player vocab : {len(player_vocab) - 1:,} ({missing} without career stats)")
    print(f"  venue vocab  : {len(venue_vocab) - 1}")
    print(f"  test seasons : {', '.join(TEST_SEASONS)}")
    print(f"  splits       : train={is_train.sum()}  "
          f"val={is_val.sum()}  test={is_test.sum()}  "
          f"(val ≈ {VAL_FRAC:.0%} per-season from train)")

    _banner("Phase 1 · PlayerEncoder (PyTorch)", char="-")
    t1 = time.perf_counter()
    player_emb, venue_emb, stage1_history = train_player_encoder(
        df, player_vocab, venue_vocab, std_stats,
        train_mask=is_train, val_mask=is_val,
    )
    print(f"  phase 1 elapsed: {time.perf_counter() - t1:.1f}s")

    _banner("Phase 2 · LightGBM on pooled features", char="-")
    t2 = time.perf_counter()
    ds = build_features(df, player_vocab, player_emb, venue_vocab, venue_emb, raw_stats)
    print(f"  feature matrix: {ds.X.shape[0]:,} rows × {ds.X.shape[1]} cols")
    X_tr, y_tr = ds.X[is_train.to_numpy()], ds.y[is_train.to_numpy()]
    X_val, y_val = ds.X[is_val.to_numpy()], ds.y[is_val.to_numpy()]
    X_te = ds.X[is_test.to_numpy()]
    y_te = ds.y[is_test.to_numpy()].reset_index(drop=True)
    seasons_te = ds.seasons[is_test.to_numpy()].reset_index(drop=True)

    model = train_model(X_tr, y_tr, X_val, y_val, categorical_cols=ds.categorical_cols)
    print(f"  phase 2 elapsed: {time.perf_counter() - t2:.1f}s  "
          f"(best_iter={model.best_iteration})")

    _banner("Results", char="-")
    baseline = season_mean_baseline(df[is_train_full], df[is_test])
    val_metrics = evaluate(model, X_val, y_val)
    test_metrics = evaluate(model, X_te, y_te)
    season_metrics = evaluate_by_season(model, X_te, y_te, seasons_te)

    # Overall baseline per season (both test seasons not in train → predict overall mean)
    overall_mean = df[is_train_full][TARGET].mean()
    baseline_by_season: dict[str, dict] = {}
    for season, grp in df[is_test].groupby("season", observed=True):
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

    pred_te = model.predict(X_te, num_iteration=model.best_iteration)
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

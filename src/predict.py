"""Inference pipeline for 1st-innings score prediction on new matches.

Usage
-----
    python predict.py --train                # train full pipeline, pickle artifacts
    python predict.py path/to/matches.csv    # predict from a CSV of new matches

As a module:
    from predict import predict, predict_one, load_artifacts
    pred = predict_one({...one match...})

Input columns (one row per match, always innings 1)
---------------------------------------------------
Required: match_id, season, venue, city, batting_team, bowling_team,
          won_toss, toss_decision, players, bowlers, date
Optional: year (derived from season if absent), team_runs (ignored)

`players` / `bowlers` are pipe-separated player names (same spelling as
in archive/matches_aggregated.csv). Unknown names are silently dropped
by PlayerVocab — a warning is printed listing them.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# modelLogic prints unicode sparkline chars; the default Windows console codec
# (cp1252) can't encode them. Reconfigure early so imports that print at load
# time don't crash either.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from modelLogic import (
    ARCHIVE,
    REMAINING_CATS,
    TARGET,
    Ensemble,
    PlayerVocab,
    ScalarVocab,
    SeasonBaseline,
    build_features,
    build_rolling_form,
    fit_season_baseline,
    load_data,
    load_player_stats,
    time_split_mask,
    train_ensemble,
    train_player_encoder,
    train_val_split_mask,
)


ARTIFACTS_PATH = ARCHIVE / "predict_artifacts.pkl"

REQUIRED_COLS = (
    "match_id", "season", "venue", "city",
    "batting_team", "bowling_team",
    "won_toss", "toss_decision",
    "players", "bowlers", "date",
)

# Columns from the training history that build_rolling_form reads — kept
# minimal so the pickled artifact stays small.
HISTORY_COLS = [
    "match_id", "season", "year", "date",
    "batting_team", "bowling_team", "venue", TARGET,
]


@dataclass
class Artifacts:
    player_vocab: PlayerVocab
    venue_vocab: ScalarVocab
    player_emb: np.ndarray
    venue_emb: np.ndarray
    raw_stats: np.ndarray
    ensemble: Ensemble
    trend: SeasonBaseline
    history_inn1: pd.DataFrame


# ----------------------------------------------------------------- Training


def train_and_save(path: Path = ARTIFACTS_PATH) -> Artifacts:
    print(f"Training full pipeline and saving artifacts to {path}")
    t0 = time.perf_counter()

    df = load_data()
    is_train_full = time_split_mask(df)
    is_train, is_val = train_val_split_mask(df, is_train_full)

    player_vocab = PlayerVocab.from_df(df)
    venue_vocab = ScalarVocab.from_series(df[is_train]["venue"])
    raw_stats = load_player_stats(player_vocab)

    print("  Stage 1: PlayerEncoder")
    player_emb, venue_emb, _ = train_player_encoder(
        df, player_vocab, venue_vocab, train_mask=is_train, val_mask=is_val,
    )

    inn1 = (df["innings"] == 1).to_numpy()
    df_inn1 = df[inn1].reset_index(drop=True)
    is_train_inn1 = is_train.to_numpy()[inn1]
    is_val_inn1 = is_val.to_numpy()[inn1]

    rolling_form = build_rolling_form(df_inn1)
    ds = build_features(
        df_inn1, player_vocab, player_emb, venue_vocab, venue_emb, raw_stats,
        rolling_form=rolling_form,
    )

    trend = fit_season_baseline(df_inn1[is_train_inn1])
    trend_all = trend.predict(df_inn1["year"].to_numpy())
    y_detr = pd.Series(ds.y.to_numpy() - trend_all, index=ds.y.index)

    print("  Stage 2: LGBM ensemble")
    ensemble = train_ensemble(
        ds.X[is_train_inn1], y_detr[is_train_inn1],
        ds.X[is_val_inn1], y_detr[is_val_inn1],
        categorical_cols=ds.categorical_cols,
        n_models=5,
    )

    art = Artifacts(
        player_vocab=player_vocab,
        venue_vocab=venue_vocab,
        player_emb=player_emb,
        venue_emb=venue_emb,
        raw_stats=raw_stats,
        ensemble=ensemble,
        trend=trend,
        history_inn1=df_inn1[HISTORY_COLS].copy(),
    )
    with path.open("wb") as f:
        pickle.dump(art, f)
    print(f"  saved ({time.perf_counter() - t0:.1f}s)")
    return art


def load_artifacts(path: Path = ARTIFACTS_PATH) -> Artifacts:
    if not path.exists():
        raise FileNotFoundError(
            f"artifacts not found at {path} — run `python predict.py --train` first"
        )
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- Inference


def _warn_unknown_players(df: pd.DataFrame, vocab: PlayerVocab) -> None:
    unknown: set[str] = set()
    for col in ("players", "bowlers"):
        for row in df[col].dropna():
            for name in str(row).split("|"):
                if name and name not in vocab.id_of:
                    unknown.add(name)
    if unknown:
        print(f"  warning: {len(unknown)} unknown player(s) — "
              f"they contribute nothing to pooled embeddings:")
        for n in sorted(unknown):
            print(f"    - {n}")


def _prepare_new_rows(new_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in new_df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = new_df.copy()
    df["innings"] = 1
    if "year" not in df.columns:
        df["year"] = df["season"].astype(str).str.slice(0, 4).astype(int)
    df["date"] = pd.to_datetime(df["date"])
    if TARGET not in df.columns:
        df[TARGET] = np.nan
    return df


def predict(new_df: pd.DataFrame, art: Artifacts | None = None) -> np.ndarray:
    """Predict 1st-innings team totals for one or more new matches."""
    if art is None:
        art = load_artifacts()
    prepped = _prepare_new_rows(new_df)
    _warn_unknown_players(prepped, art.player_vocab)

    # Concatenate history + new rows so each new row's rolling-form
    # features see the full training history. The new rows' own team_runs
    # is NaN, and build_rolling_form uses .shift(1) so the current match
    # never feeds into its own history either way.
    hist = art.history_inn1.copy()
    combined = pd.concat(
        [hist.assign(_is_new=False),
         prepped[HISTORY_COLS].assign(_is_new=True)],
        ignore_index=True,
    )
    rf_all = build_rolling_form(combined)
    new_mask = combined["_is_new"].to_numpy()
    rf_new = rf_all[new_mask].reset_index(drop=True)

    prepped = prepped.reset_index(drop=True)
    for col in REMAINING_CATS + ["venue"]:
        prepped[col] = prepped[col].astype("category")

    ds = build_features(
        prepped, art.player_vocab, art.player_emb,
        art.venue_vocab, art.venue_emb, art.raw_stats,
        rolling_form=rf_new,
    )
    offset = art.trend.predict(prepped["year"].to_numpy())
    return art.ensemble.predict(ds.X) + offset


def predict_one(row: dict, art: Artifacts | None = None) -> float:
    """Predict a single match. `row` is a dict of the REQUIRED_COLS."""
    return float(predict(pd.DataFrame([row]), art=art)[0])


# --------------------------------------------------------------------- CLI


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Predict 1st-innings totals.")
    parser.add_argument("--train", action="store_true",
                        help="train the full pipeline and save artifacts")
    parser.add_argument("input", nargs="?",
                        help="CSV of new matches (innings-1 rows)")
    args = parser.parse_args()

    if args.train:
        train_and_save()
        return 0
    if not args.input:
        parser.error("provide an input CSV path, or pass --train")

    new_df = pd.read_csv(args.input)
    pred = predict(new_df)
    out = new_df.copy()
    out["predicted_team_runs"] = np.round(pred, 1)
    cols = [c for c in ("match_id", "season", "batting_team", "bowling_team",
                        "venue", "predicted_team_runs") if c in out.columns]
    print()
    print(out[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())

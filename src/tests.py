"""Embedding validity tests.

1. Nearest-neighbor spot checks (qualitative):
   For each well-known player, print top-k nearest neighbors in the
   learned embedding space (cosine). Good embeddings put similar
   players next to each other.

2. Stage-2 ablation (quantitative):
   Re-train the LGBM stage with (a) real, (b) random, (c) zeroed player
   embeddings while holding venue embeddings and career stats fixed.
   A meaningful gap between (a) and (b/c) means the embeddings are
   carrying signal the downstream model actually uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelLogic import (
    PlayerVocab,
    ScalarVocab,
    build_features,
    evaluate,
    load_data,
    load_player_stats,
    time_split_mask,
    train_model,
    train_player_encoder,
    train_val_split_mask,
)


SAMPLE_PLAYERS: list[str] = [
    "V Kohli",
    "RG Sharma",
    "DA Warner",
    "MS Dhoni",
    "AB de Villiers",
    "KL Rahul",
    "JJ Bumrah",
    "R Ashwin",
    "B Kumar",
    "YS Chahal",
    "Rashid Khan",
    "RA Jadeja",
    "HH Pandya",
    "AD Russell",
]


def _banner(text: str, char: str = "=") -> None:
    line = char * (len(text) + 6)
    print(f"\n{line}\n== {text} ==\n{line}")


def nearest_neighbors(
    player_emb: np.ndarray,
    names: np.ndarray,
    query_names: list[str] = SAMPLE_PLAYERS,
    k: int = 5,
) -> None:
    name_to_idx = {n: i for i, n in enumerate(names)}
    norms = np.linalg.norm(player_emb, axis=1, keepdims=True).clip(min=1e-9)
    unit = player_emb / norms
    width = max(len(q) for q in query_names)

    for q in query_names:
        if q not in name_to_idx:
            print(f"  {q:<{width}}  ->  (not in vocab)")
            continue
        idx = name_to_idx[q]
        if np.linalg.norm(player_emb[idx]) < 1e-9:
            print(f"  {q:<{width}}  ->  (zero embedding, no training signal)")
            continue
        sim = unit @ unit[idx]
        # Exclude self and pad row.
        sim[idx] = -np.inf
        sim[0] = -np.inf
        top = np.argsort(-sim)[:k]
        neighbors = ", ".join(f"{names[j]} ({sim[j]:+.2f})" for j in top)
        print(f"  {q:<{width}}  ->  {neighbors}")


def ablate_player_emb(
    player_emb: np.ndarray,
    venue_emb: np.ndarray,
    df_inn1: pd.DataFrame,
    player_vocab: PlayerVocab,
    venue_vocab: ScalarVocab,
    raw_stats: np.ndarray,
    is_train_inn1: np.ndarray,
    is_val_inn1: np.ndarray,
    is_test_inn1: np.ndarray,
    drop_stats: bool = False,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(0)
    scale = float(player_emb.std())
    variants: dict[str, np.ndarray] = {
        "real":   player_emb,
        "random": (rng.standard_normal(player_emb.shape) * scale).astype(np.float32),
        "zero":   np.zeros_like(player_emb),
    }
    results: dict[str, dict[str, float]] = {}
    for name, emb in variants.items():
        print(f"\n  --- variant: {name} ---")
        ds = build_features(df_inn1, player_vocab, emb, venue_vocab, venue_emb, raw_stats)
        X = ds.X
        if drop_stats:
            stat_cols = [c for c in X.columns if "_stat_" in c]
            X = X.drop(columns=stat_cols)
            if name == "real":
                print(f"    dropped {len(stat_cols)} career-stat columns "
                      f"(feature matrix: {X.shape[1]} cols)")
        X_tr, y_tr = X[is_train_inn1], ds.y[is_train_inn1]
        X_val, y_val = X[is_val_inn1], ds.y[is_val_inn1]
        X_te = X[is_test_inn1]
        y_te = ds.y[is_test_inn1].reset_index(drop=True)
        model = train_model(X_tr, y_tr, X_val, y_val, categorical_cols=ds.categorical_cols)
        results[name] = evaluate(model, X_te, y_te)
    return results


def main() -> None:
    _banner("Embedding validity tests")

    _banner("Phase 0  loading data and training Stage 1", char="-")
    df = load_data()
    is_train_full = time_split_mask(df)
    is_train, is_val = train_val_split_mask(df, is_train_full)
    is_test = ~is_train_full

    player_vocab = PlayerVocab.from_df(df)
    venue_vocab = ScalarVocab.from_series(df[is_train]["venue"])
    raw_stats = load_player_stats(player_vocab)

    player_emb, venue_emb, _ = train_player_encoder(
        df, player_vocab, venue_vocab,
        train_mask=is_train, val_mask=is_val,
    )

    names = np.empty(len(player_vocab), dtype=object)
    for n, i in player_vocab.id_of.items():
        names[i] = n

    _banner("Test 1  nearest-neighbor spot checks", char="-")
    nearest_neighbors(player_emb, names)

    _banner("Test 2  Stage-2 ablation (NO career stats)", char="-")
    inn1 = (df["innings"] == 1).to_numpy()
    df_inn1 = df[inn1].reset_index(drop=True)
    is_train_inn1 = is_train.to_numpy()[inn1]
    is_val_inn1 = is_val.to_numpy()[inn1]
    is_test_inn1 = is_test.to_numpy()[inn1]

    results = ablate_player_emb(
        player_emb, venue_emb, df_inn1,
        player_vocab, venue_vocab, raw_stats,
        is_train_inn1, is_val_inn1, is_test_inn1,
        drop_stats=True,
    )

    _banner("Ablation summary, no stats (test set)", char="-")
    real_mae = results["real"]["MAE"]
    print(f"  {'variant':>8s}   {'MAE':>7s}   {'RMSE':>7s}   {'dMAE':>7s}")
    print(f"  {'-' * 8}   {'-' * 7}   {'-' * 7}   {'-' * 7}")
    for name, m in results.items():
        delta = m["MAE"] - real_mae
        print(f"  {name:>8s}   {m['MAE']:7.2f}   {m['RMSE']:7.2f}   {delta:+7.2f}")


if __name__ == "__main__":
    main()

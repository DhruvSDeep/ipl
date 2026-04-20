"""Pre-match IPL 1st-innings score prediction.

Two-stage architecture
----------------------
Stage 1 -- PlayerEncoder (PyTorch)
    Each player is represented by a *hybrid token*:
        token = [ learned_embedding (16)  ||  static_career_stats (5) ]
    The stats come from `player_stats.csv` (runs, strike_rate, wickets,
    balls_bowled, economy) standardised column-wise. Tokens are mean-pooled
    across each match's batter list and bowler list, then concatenated with
    a venue embedding (8) and one-hots of the remaining categoricals, and
    passed through Linear -> ReLU -> Dropout -> Linear -> scalar.

    The stats tensor is registered as a buffer (not a parameter), so only
    the embedding tables and MLP weights learn; stats stay fixed.

Stage 2 -- LGBM regressor
    Consumes the remaining raw categoricals plus, per match:
      - mean batter embedding     (16)
      - mean bowler embedding     (16)
      - mean batter career stats  (5, raw values for interpretability)
      - mean bowler career stats  (5)
      - venue embedding           (8)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error


ARCHIVE = Path(__file__).resolve().parent.parent / "archive"
DATA_PATH = ARCHIVE / "matches_aggregated.csv"
PLAYER_STATS_PATH = ARCHIVE / "player_stats.csv"
LOSS_PLOT_PATH = ARCHIVE / "stage1_loss.png"


def _sparkline(values: list[float]) -> str:
    bars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi == lo:
        return bars[0] * len(values)
    return "".join(bars[int((v - lo) / (hi - lo) * (len(bars) - 1))] for v in values)


PLAYER_STAT_COLS = ["runs", "strike_rate", "wickets", "balls_bowled", "economy"]

EMBEDDED_SCALAR_CATS: dict[str, int] = {"venue": 8}

REMAINING_CATS: list[str] = [
    "season",
    "city",
    "inn1_won_toss",
]

PLAYER_LIST_COLS = {
    "bat": "inn1_players",
    "bowl": "inn1_opp_bowlers",
}

TARGET = "inn1_runs"
TEST_SEASONS = ("2017", "2025")
VAL_FRAC = 0.05
VAL_SEED = 42

EMBED_DIM = 16
EMBED_EPOCHS = 80
EMBED_LR = 5e-3
EMBED_BATCH = 64
EMBED_PATIENCE = 10

LGBM_PARAMS: dict = {
    "objective": "regression_l1",
    "metric": ["mae", "rmse"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbose": -1,
}


# ---------------------------------------------------------------- Data loading


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["inn1_won_toss"] = (df["toss_winner"] == df["inn1_batting_team"]).astype("int8")
    for col in REMAINING_CATS + list(EMBEDDED_SCALAR_CATS):
        df[col] = df[col].astype("category")
    return df


# -------------------------------------------------------------- Vocabularies


@dataclass
class PlayerVocab:
    id_of: dict[str, int]
    PAD: int = 0

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "PlayerVocab":
        names: set[str] = set()
        for col in PLAYER_LIST_COLS.values():
            for row in df[col].dropna():
                names.update(row.split("|"))
        id_of = {"<pad>": 0}
        for n in sorted(names):
            id_of[n] = len(id_of)
        return cls(id_of=id_of)

    def __len__(self) -> int:
        return len(self.id_of)

    def encode(self, pipe_str: str | float) -> list[int]:
        if not isinstance(pipe_str, str):
            return []
        return [self.id_of[p] for p in pipe_str.split("|") if p in self.id_of]


@dataclass
class ScalarVocab:
    id_of: dict[str, int]
    UNK: int = 0

    @classmethod
    def from_series(cls, s: pd.Series) -> "ScalarVocab":
        id_of = {"<unk>": 0}
        for v in sorted(s.dropna().astype(str).unique()):
            id_of[v] = len(id_of)
        return cls(id_of=id_of)

    def __len__(self) -> int:
        return len(self.id_of)

    def encode_series(self, s: pd.Series) -> np.ndarray:
        return s.astype(str).map(self.id_of).fillna(self.UNK).astype(np.int64).to_numpy()


# ---------------------------------------------------- Per-player stats tensor


def load_player_stats(
    vocab: PlayerVocab, path: Path = PLAYER_STATS_PATH
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (vocab_size, len(STAT_COLS)) matrix indexed by player id.

    Returns (raw, standardized). Row 0 (pad) and any vocab player missing
    from the stats file stay at zero in both versions.
    """
    stats_df = pd.read_csv(path).set_index("player")
    raw = np.zeros((len(vocab), len(PLAYER_STAT_COLS)), dtype=np.float32)
    for name, idx in vocab.id_of.items():
        if name in stats_df.index:
            raw[idx] = stats_df.loc[name, PLAYER_STAT_COLS].to_numpy(dtype=np.float32)

    active = raw[1:]  # drop pad row from normalisation stats
    mean = active.mean(axis=0)
    std = active.std(axis=0)
    std[std == 0] = 1.0
    std_mat = raw.copy()
    std_mat[1:] = (active - mean) / std
    return raw, std_mat


# -------------------------------------------------------- Stage 1: embedding NN


def _pad(seqs: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    L = max((len(s) for s in seqs), default=1)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.float32)
    for i, s in enumerate(seqs):
        if s:
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1.0
    return ids, mask


class EmbedDataset(TorchDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        player_vocab: PlayerVocab,
        venue_vocab: ScalarVocab,
        cat_onehot: np.ndarray,
    ):
        self.bat = [player_vocab.encode(s) for s in df[PLAYER_LIST_COLS["bat"]]]
        self.bowl = [player_vocab.encode(s) for s in df[PLAYER_LIST_COLS["bowl"]]]
        self.venue = torch.tensor(venue_vocab.encode_series(df["venue"]), dtype=torch.long)
        self.cat = torch.tensor(cat_onehot, dtype=torch.float32)
        self.y = torch.tensor(df[TARGET].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.bat[i], self.bowl[i], self.venue[i], self.cat[i], self.y[i]


def _collate(batch):
    bat, bowl, venue, cat, y = zip(*batch)
    bat_ids, bat_mask = _pad(list(bat))
    bowl_ids, bowl_mask = _pad(list(bowl))
    return (
        bat_ids, bat_mask,
        bowl_ids, bowl_mask,
        torch.stack(venue),
        torch.stack(cat),
        torch.stack(y),
    )


class PlayerEncoder(nn.Module):
    """Hybrid player token = [learned embedding || fixed career-stats vector]."""

    def __init__(
        self,
        player_vocab_size: int,
        venue_vocab_size: int,
        cat_dim: int,
        stats_matrix: np.ndarray,
        embed_dim: int = EMBED_DIM,
        venue_dim: int = EMBEDDED_SCALAR_CATS["venue"],
    ):
        super().__init__()
        self.player_embed = nn.Embedding(player_vocab_size, embed_dim, padding_idx=0)
        self.venue_embed = nn.Embedding(venue_vocab_size, venue_dim, padding_idx=0)

        # Career stats are fixed. Register as buffer so they move with .to(device).
        self.register_buffer("player_stats", torch.from_numpy(stats_matrix).float())
        stats_dim = stats_matrix.shape[1]
        player_token_dim = embed_dim + stats_dim

        hidden = 64
        self.head = nn.Sequential(
            nn.Linear(2 * player_token_dim + venue_dim + cat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def _player_token(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.player_embed(ids), self.player_stats[ids]], dim=-1)

    @staticmethod
    def _masked_mean(tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (tok * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, bat_ids, bat_mask, bowl_ids, bowl_mask, venue_ids, cat):
        bat_vec = self._masked_mean(self._player_token(bat_ids), bat_mask)
        bowl_vec = self._masked_mean(self._player_token(bowl_ids), bowl_mask)
        venue_vec = self.venue_embed(venue_ids)
        x = torch.cat([bat_vec, bowl_vec, venue_vec, cat], dim=-1)
        return self.head(x).squeeze(-1)


def _onehot_remaining_cats(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    dummies = pd.get_dummies(df[REMAINING_CATS], drop_first=False)
    return dummies.to_numpy(dtype=np.float32), dummies.columns.tolist()


def _eval_mae(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            *inputs, y = batch
            pred = model(*inputs)
            total += loss_fn(pred, y).item() * y.size(0)
            n += y.size(0)
    return total / n


def train_player_encoder(
    df: pd.DataFrame,
    player_vocab: PlayerVocab,
    venue_vocab: ScalarVocab,
    stats_matrix: np.ndarray,
    train_mask: pd.Series,
    val_mask: pd.Series,
    epochs: int = EMBED_EPOCHS,
    batch_size: int = EMBED_BATCH,
    lr: float = EMBED_LR,
    embed_dim: int = EMBED_DIM,
    patience: int = EMBED_PATIENCE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train Stage 1 with val-based early stopping.

    Stops after `patience` consecutive epochs with no improvement in
    validation MAE and restores the best-val weights before returning.
    Returns (player_emb, venue_emb, history) where history has keys
    "train", "val", "best_epoch".
    """
    cat_mat, _ = _onehot_remaining_cats(df)
    train_rows = df[train_mask].reset_index(drop=True)
    val_rows = df[val_mask].reset_index(drop=True)
    train_ds = EmbedDataset(train_rows, player_vocab, venue_vocab, cat_mat[train_mask.to_numpy()])
    val_ds = EmbedDataset(val_rows, player_vocab, venue_vocab, cat_mat[val_mask.to_numpy()])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    model = PlayerEncoder(
        player_vocab_size=len(player_vocab),
        venue_vocab_size=len(venue_vocab),
        cat_dim=cat_mat.shape[1],
        stats_matrix=stats_matrix,
        embed_dim=embed_dim,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  PlayerEncoder: {n_params:,} trainable params "
          f"(vocab {len(player_vocab)}, venues {len(venue_vocab)}, cat_dim {cat_mat.shape[1]})")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.L1Loss()

    train_hist: list[float] = []
    val_hist: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    bad_epochs = 0
    stopped_early = False

    pbar = tqdm(range(epochs), desc="  training", unit="ep", leave=True)
    for epoch in pbar:
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            *inputs, y = batch
            pred = model(*inputs)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
            n += y.size(0)
        train_mae = total / n
        val_mae = _eval_mae(model, val_loader, loss_fn)
        train_hist.append(train_mae)
        val_hist.append(val_mae)
        pbar.set_postfix(train=f"{train_mae:.2f}", val=f"{val_mae:.2f}")

        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                stopped_early = True
                pbar.close()
                print(f"  early stop at epoch {epoch + 1} "
                      f"(no val improvement for {patience} epochs; "
                      f"best val MAE {best_val:.2f} @ epoch {best_epoch + 1})")
                break

    if not stopped_early:
        print(f"  ran full {epochs} epochs  "
              f"(best val MAE {best_val:.2f} @ epoch {best_epoch + 1})")

    if best_state is not None:
        model.load_state_dict(best_state)

    if len(val_hist) >= 2:
        print(f"  train {_sparkline(train_hist)}  [{train_hist[0]:.1f} → {train_hist[-1]:.1f}]")
        print(f"  val   {_sparkline(val_hist)}  [{val_hist[0]:.1f} → {val_hist[-1]:.1f}]")

    with torch.no_grad():
        player_emb = model.player_embed.weight.detach().cpu().numpy()
        venue_emb = model.venue_embed.weight.detach().cpu().numpy()
    history = {"train": train_hist, "val": val_hist, "best_epoch": best_epoch}
    return player_emb, venue_emb, history


def pool_by_id(
    pipe_series: pd.Series, vocab: PlayerVocab, matrix: np.ndarray
) -> np.ndarray:
    """Mean of `matrix[ids]` per match; shape (n_matches, matrix.shape[1])."""
    out = np.zeros((len(pipe_series), matrix.shape[1]), dtype=np.float32)
    for i, s in enumerate(pipe_series):
        ids = vocab.encode(s)
        if ids:
            out[i] = matrix[ids].mean(axis=0)
    return out


def lookup_venue_embedding(
    series: pd.Series, vocab: ScalarVocab, embeddings: np.ndarray
) -> np.ndarray:
    ids = vocab.encode_series(series)
    return embeddings[ids]


# -------------------------------------------------------- Stage 2: LGBM wrapper


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    seasons: pd.Series
    categorical_cols: list[str] = field(default_factory=list)


def build_features(
    df: pd.DataFrame,
    player_vocab: PlayerVocab,
    player_emb: np.ndarray,
    venue_vocab: ScalarVocab,
    venue_emb: np.ndarray,
    player_stats_raw: np.ndarray,
) -> Dataset:
    parts: list[pd.DataFrame] = [df[REMAINING_CATS].reset_index(drop=True)]

    for prefix, col in PLAYER_LIST_COLS.items():
        pooled_emb = pool_by_id(df[col], player_vocab, player_emb)
        parts.append(pd.DataFrame(
            pooled_emb,
            columns=[f"{prefix}_emb_{i}" for i in range(pooled_emb.shape[1])],
        ))
        pooled_stat = pool_by_id(df[col], player_vocab, player_stats_raw)
        parts.append(pd.DataFrame(
            pooled_stat,
            columns=[f"{prefix}_stat_{c}" for c in PLAYER_STAT_COLS],
        ))

    venue_vec = lookup_venue_embedding(df["venue"], venue_vocab, venue_emb)
    parts.append(pd.DataFrame(
        venue_vec,
        columns=[f"venue_emb_{i}" for i in range(venue_vec.shape[1])],
    ))

    X = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    y = df[TARGET].reset_index(drop=True)
    seasons = df["season"].astype(str).reset_index(drop=True)
    return Dataset(X=X, y=y, seasons=seasons, categorical_cols=REMAINING_CATS)


def time_split_mask(df: pd.DataFrame, test_seasons: tuple[str, ...] = TEST_SEASONS) -> pd.Series:
    return ~df["season"].astype(str).isin(test_seasons)


def train_val_split_mask(
    df: pd.DataFrame,
    train_full_mask: pd.Series,
    frac: float = VAL_FRAC,
    seed: int = VAL_SEED,
) -> tuple[pd.Series, pd.Series]:
    """Stratify `frac` of each season's train rows into validation.

    Returns (train_mask, val_mask) over the full df index, disjoint from
    the test set (rows where train_full_mask is False).
    """
    rng = np.random.default_rng(seed)
    val_idx: list = []
    for _, grp in df[train_full_mask].groupby("season", observed=True):
        n = max(1, int(round(len(grp) * frac)))
        val_idx.extend(rng.choice(grp.index, size=n, replace=False).tolist())
    val_mask = pd.Series(False, index=df.index)
    val_mask.loc[val_idx] = True
    return (train_full_mask & ~val_mask), val_mask


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_cols: list[str],
    params: dict | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
) -> lgb.Booster:
    params = {**LGBM_PARAMS, **(params or {})}
    train_set = lgb.Dataset(X_train, y_train, categorical_feature=categorical_cols)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=categorical_cols)
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(100)],
    )


def evaluate(model: lgb.Booster, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = model.predict(X, num_iteration=model.best_iteration)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
    }


def evaluate_by_season(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
) -> dict[str, dict[str, float]]:
    pred = model.predict(X, num_iteration=model.best_iteration)
    y_arr = y.to_numpy()
    s_arr = seasons.to_numpy()
    results: dict[str, dict[str, float]] = {}
    for season in sorted(np.unique(s_arr)):
        mask = s_arr == season
        results[str(season)] = {
            "MAE": float(mean_absolute_error(y_arr[mask], pred[mask])),
            "RMSE": float(np.sqrt(mean_squared_error(y_arr[mask], pred[mask]))),
            "n": int(mask.sum()),
        }
    return results


def season_mean_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    season_mean = train_df.groupby("season", observed=True)[TARGET].mean()
    overall_mean = train_df[TARGET].mean()
    pred = test_df["season"].map(season_mean).fillna(overall_mean)
    return {
        "MAE": float(mean_absolute_error(test_df[TARGET], pred)),
        "RMSE": float(np.sqrt(mean_squared_error(test_df[TARGET], pred))),
    }



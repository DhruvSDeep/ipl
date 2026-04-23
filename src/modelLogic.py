"""Pre-match IPL 1st-innings score prediction.

Two-stage architecture
----------------------
Stage 1 -- PlayerEncoder (PyTorch, multi-task)
    Each player is a learned 16-d embedding. Three prediction heads
    operate on these embeddings:

        team_head    : pooled(batters, bowlers) + venue + cats   →  team_runs
        batter_head  : own token + pooled opposition bowlers     →  runs_scored
        bowler_head  : own token + pooled opposition batters     →  runs_conceded

    Losses:
        team_loss   = MAE on innings==1 rows only (chase-truncation makes
                      2nd-innings totals noisy for pre-match prediction)
        batter_loss = masked MAE over players with did_bat == 1
        bowler_loss = masked MAE over players with did_bowl == 1
        total       = team_loss + λ_b * batter_loss + λ_w * bowler_loss

    The per-player heads condition on the opposition pool so a batter's
    predicted runs can depend on the bowling attack they face (and vice
    versa), pushing matchup signal into the embeddings.

Stage 2 -- LGBM regressor (innings==1 rows only)
    Consumes raw categoricals plus, per match:
      - mean batter embedding     (16)
      - mean bowler embedding     (16)
      - mean batter career stats  (5)
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
BALL_BY_BALL_PATH = ARCHIVE / "IPL.csv"
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
    "won_toss",
    "batting_team",
    "bowling_team",
    "toss_decision",
]

# Numeric, non-categorical features. Kept separate from REMAINING_CATS so LGBM
# treats them as continuous (enabling extrapolation to unseen values — the
# `season` category 2024/2025 is unseen at train time, but the `year` value
# 2024 is just a bigger number than the max seen in train).
NUMERIC_FEATS: list[str] = ["year"]

PLAYER_LIST_COLS = {"bat": "players", "bowl": "bowlers"}
PER_PLAYER_RUNS = {"bat": "runs_scored", "bowl": "runs_conceded"}
PER_PLAYER_DID = {"bat": "did_bat", "bowl": "did_bowl"}

TARGET = "team_runs"
# Latest two seasons reserved as a genuine future holdout.
TEST_SEASONS = ("2024", "2025")
VAL_FRAC = 0.15
VAL_SEED = 42

EMBED_DIM = 16
EMBED_EPOCHS = 120
EMBED_LR = 5e-3
EMBED_BATCH = 64
EMBED_PATIENCE = 25

# Auxiliary-head loss weights. Team-total loss is 1.0 by convention.
BAT_LOSS_WEIGHT = 0.1
BOWL_LOSS_WEIGHT = 0.1

# Rolling-form window sizes. Batting/bowling team history is sparse (a team
# appears as batting_team in maybe half of its matches), so use short windows
# to capture real signal without needing deep history.
BAT_ROLL_WINDOWS = (3, 5, 10)
BOWL_ROLL_WINDOWS = (3, 5, 10)
VENUE_ROLL_WINDOWS = (5, 10, 20)

# Recency weighting for season-baseline extrapolation. Small τ → steeper curve
# fitted mostly to the last 3-5 seasons (captures post-2022 scoring jump).
SEASON_TREND_TAU = 2.0

LGBM_PARAMS: dict = {
    "objective": "huber",
    "alpha": 1.35,
    "metric": ["mae", "rmse"],
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 12,
    "lambda_l2": 1.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


# ---------------------------------------------------------------- Data loading


def load_data(
    path: Path = DATA_PATH, ball_path: Path = BALL_BY_BALL_PATH
) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Numeric year, e.g. "2020/21" → 2020. Lets tree models split on time
    # order and extrapolate past the max season seen in training.
    df["year"] = df["season"].astype(str).str.slice(0, 4).astype(int)
    # Real match date from ball-by-ball CSV — match_id alone isn't strictly
    # chronological (cricsheet assigns IDs at ingest time), so we need dates
    # for leak-free rolling-form features.
    dates = (
        pd.read_csv(ball_path, usecols=["match_id", "date"])
        .drop_duplicates("match_id")
    )
    dates["date"] = pd.to_datetime(dates["date"])
    df = df.merge(dates, on="match_id", how="left")
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
) -> np.ndarray:
    """Build a (vocab_size, len(STAT_COLS)) matrix indexed by player id.

    Row 0 (pad) and any vocab player missing from the stats file stay at
    zero. Used by Stage 2 — Stage 1 learns embeddings from scratch.
    """
    stats_df = pd.read_csv(path).set_index("player")
    raw = np.zeros((len(vocab), len(PLAYER_STAT_COLS)), dtype=np.float32)
    for name, idx in vocab.id_of.items():
        if name in stats_df.index:
            raw[idx] = stats_df.loc[name, PLAYER_STAT_COLS].to_numpy(dtype=np.float32)
    return raw


# -------------------------------------------------------- Stage 1: embedding NN


def _parse_pipe_floats(series: pd.Series) -> list[list[float]]:
    out: list[list[float]] = []
    for s in series:
        if not isinstance(s, str) or not s:
            out.append([])
        else:
            out.append([float(v) for v in s.split("|")])
    return out


def _pad_ids(seqs: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    L = max((len(s) for s in seqs), default=1)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.float32)
    for i, s in enumerate(seqs):
        if s:
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1.0
    return ids, mask


def _pad_floats(seqs: list[list[float]], L: int) -> torch.Tensor:
    out = torch.zeros((len(seqs), L), dtype=torch.float32)
    for i, s in enumerate(seqs):
        if s:
            out[i, : len(s)] = torch.tensor(s, dtype=torch.float32)
    return out


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
        self.bat_runs = _parse_pipe_floats(df[PER_PLAYER_RUNS["bat"]])
        self.bowl_runs = _parse_pipe_floats(df[PER_PLAYER_RUNS["bowl"]])
        self.bat_did = _parse_pipe_floats(df[PER_PLAYER_DID["bat"]])
        self.bowl_did = _parse_pipe_floats(df[PER_PLAYER_DID["bowl"]])
        self.venue = torch.tensor(venue_vocab.encode_series(df["venue"]), dtype=torch.long)
        self.cat = torch.tensor(cat_onehot, dtype=torch.float32)
        self.y = torch.tensor(df[TARGET].to_numpy(), dtype=torch.float32)
        # Team-total loss only makes sense on 1st-innings rows (chase-truncated totals
        # in innings 2 are not well-defined pre-match targets).
        self.is_inn1 = torch.tensor(
            (df["innings"].to_numpy() == 1).astype(np.float32), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return (
            self.bat[i], self.bat_runs[i], self.bat_did[i],
            self.bowl[i], self.bowl_runs[i], self.bowl_did[i],
            self.venue[i], self.cat[i], self.y[i], self.is_inn1[i],
        )


def _collate(batch):
    (bat, bat_runs, bat_did,
     bowl, bowl_runs, bowl_did,
     venue, cat, y, is_inn1) = zip(*batch)

    bat_ids, bat_pad = _pad_ids(list(bat))
    bowl_ids, bowl_pad = _pad_ids(list(bowl))
    L_bat, L_bowl = bat_ids.size(1), bowl_ids.size(1)

    bat_runs_t = _pad_floats(list(bat_runs), L_bat)
    bowl_runs_t = _pad_floats(list(bowl_runs), L_bowl)
    bat_did_t = _pad_floats(list(bat_did), L_bat)
    bowl_did_t = _pad_floats(list(bowl_did), L_bowl)

    # Loss mask = padding mask AND actual-participation mask
    bat_loss_mask = bat_pad * bat_did_t
    bowl_loss_mask = bowl_pad * bowl_did_t

    return (
        bat_ids, bat_pad, bat_runs_t, bat_loss_mask,
        bowl_ids, bowl_pad, bowl_runs_t, bowl_loss_mask,
        torch.stack(venue), torch.stack(cat),
        torch.stack(y), torch.stack(is_inn1),
    )


class PlayerEncoder(nn.Module):
    """Learned player embeddings with three prediction heads.

    team_head    : pooled(batters, bowlers) + venue + cats → team runs
    batter_head  : own token + pooled opposition bowlers   → runs_scored
    bowler_head  : own token + pooled opposition batters   → runs_conceded
    """

    def __init__(
        self,
        player_vocab_size: int,
        venue_vocab_size: int,
        cat_dim: int,
        embed_dim: int = EMBED_DIM,
        venue_dim: int = EMBEDDED_SCALAR_CATS["venue"],
    ):
        super().__init__()
        self.player_embed = nn.Embedding(player_vocab_size, embed_dim, padding_idx=0)
        self.venue_embed = nn.Embedding(venue_vocab_size, venue_dim, padding_idx=0)

        hidden = 64
        # Team head sees mean + max pool of batters and bowlers (4 * embed_dim)
        # — mean alone washes out a single star in an otherwise average XI.
        self.team_head = nn.Sequential(
            nn.Linear(4 * embed_dim + venue_dim + cat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )
        self.batter_head = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.bowler_head = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _masked_mean(tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (tok * mask.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _masked_max(tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(tok.dtype).min
        masked = tok.masked_fill(mask.unsqueeze(-1) == 0, neg_inf)
        return masked.max(dim=1).values

    def forward(self, bat_ids, bat_pad, bowl_ids, bowl_pad, venue_ids, cat):
        bat_tok = self.player_embed(bat_ids)
        bowl_tok = self.player_embed(bowl_ids)
        bat_mean = self._masked_mean(bat_tok, bat_pad)
        bowl_mean = self._masked_mean(bowl_tok, bowl_pad)
        bat_max = self._masked_max(bat_tok, bat_pad)
        bowl_max = self._masked_max(bowl_tok, bowl_pad)
        venue_vec = self.venue_embed(venue_ids)
        team = self.team_head(
            torch.cat([bat_mean, bat_max, bowl_mean, bowl_max, venue_vec, cat], dim=-1)
        ).squeeze(-1)

        # Per-player heads keep mean-only opposition context (no dim cascade).
        bowl_ctx = bowl_mean.unsqueeze(1).expand(-1, bat_tok.size(1), -1)
        bat_pred = self.batter_head(torch.cat([bat_tok, bowl_ctx], dim=-1)).squeeze(-1)

        bat_ctx = bat_mean.unsqueeze(1).expand(-1, bowl_tok.size(1), -1)
        bowl_pred = self.bowler_head(torch.cat([bowl_tok, bat_ctx], dim=-1)).squeeze(-1)

        return team, bat_pred, bowl_pred


def _onehot_remaining_cats(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    dummies = pd.get_dummies(df[REMAINING_CATS], drop_first=False)
    # Append standardised numeric features so they flow into Stage 1's
    # team_head alongside the one-hots.
    num = df[NUMERIC_FEATS].astype(np.float32).to_numpy()
    num = (num - num.mean(axis=0)) / (num.std(axis=0) + 1e-6)
    cat = dummies.to_numpy(dtype=np.float32)
    combined = np.concatenate([cat, num], axis=1)
    return combined, dummies.columns.tolist() + NUMERIC_FEATS


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Mean L1 over positions where mask == 1. Returns (loss, n_active)."""
    n = mask.sum()
    if n.item() == 0:
        return pred.new_zeros(()), 0.0
    err = (pred - target).abs() * mask
    return err.sum() / n, n.item()


def _eval_losses(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    tot = {"team": 0.0, "bat": 0.0, "bowl": 0.0}
    cnt = {"team": 0.0, "bat": 0.0, "bowl": 0.0}
    with torch.no_grad():
        for batch in loader:
            (bat_ids, bat_pad, bat_runs, bat_mask,
             bowl_ids, bowl_pad, bowl_runs, bowl_mask,
             venue_ids, cat, y, is_inn1) = batch
            team_pred, bat_pred, bowl_pred = model(
                bat_ids, bat_pad, bowl_ids, bowl_pad, venue_ids, cat
            )
            n_inn1 = is_inn1.sum().item()
            if n_inn1 > 0:
                tot["team"] += ((team_pred - y).abs() * is_inn1).sum().item()
                cnt["team"] += n_inn1
            bat_n = bat_mask.sum().item()
            if bat_n > 0:
                tot["bat"] += ((bat_pred - bat_runs).abs() * bat_mask).sum().item()
                cnt["bat"] += bat_n
            bowl_n = bowl_mask.sum().item()
            if bowl_n > 0:
                tot["bowl"] += ((bowl_pred - bowl_runs).abs() * bowl_mask).sum().item()
                cnt["bowl"] += bowl_n
    return {k: (tot[k] / cnt[k]) if cnt[k] > 0 else float("nan") for k in tot}


def train_player_encoder(
    df: pd.DataFrame,
    player_vocab: PlayerVocab,
    venue_vocab: ScalarVocab,
    train_mask: pd.Series,
    val_mask: pd.Series,
    epochs: int = EMBED_EPOCHS,
    batch_size: int = EMBED_BATCH,
    lr: float = EMBED_LR,
    embed_dim: int = EMBED_DIM,
    patience: int = EMBED_PATIENCE,
    bat_weight: float = BAT_LOSS_WEIGHT,
    bowl_weight: float = BOWL_LOSS_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train Stage 1 with val-based early stopping on the team-total MAE.

    The per-player heads are auxiliary: they push signal into the embeddings
    but early stopping tracks team MAE, the primary Stage-2 target.
    """
    # Deterministic stage-1 training so stage-2 sees stable embeddings
    # (LGBM is sensitive to the exact numeric values of pooled features).
    torch.manual_seed(VAL_SEED)
    np.random.seed(VAL_SEED)
    cat_mat, _ = _onehot_remaining_cats(df)
    train_rows = df[train_mask].reset_index(drop=True)
    val_rows = df[val_mask].reset_index(drop=True)
    train_ds = EmbedDataset(train_rows, player_vocab, venue_vocab, cat_mat[train_mask.to_numpy()])
    val_ds = EmbedDataset(val_rows, player_vocab, venue_vocab, cat_mat[val_mask.to_numpy()])
    g = torch.Generator()
    g.manual_seed(VAL_SEED)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    model = PlayerEncoder(
        player_vocab_size=len(player_vocab),
        venue_vocab_size=len(venue_vocab),
        cat_dim=cat_mat.shape[1],
        embed_dim=embed_dim,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  PlayerEncoder: {n_params:,} trainable params "
          f"(vocab {len(player_vocab)}, venues {len(venue_vocab)}, cat_dim {cat_mat.shape[1]})")
    print(f"  loss = team + {bat_weight:.2f}·batter + {bowl_weight:.2f}·bowler "
          f"(team head masked to innings==1)")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_hist: list[float] = []
    val_hist: list[float] = []
    val_bat_hist: list[float] = []
    val_bowl_hist: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    bad_epochs = 0
    stopped_early = False

    pbar = tqdm(range(epochs), desc="  training", unit="ep", leave=True)
    for epoch in pbar:
        model.train()
        tot_team = 0.0
        cnt_team = 0.0
        for batch in train_loader:
            (bat_ids, bat_pad, bat_runs, bat_mask,
             bowl_ids, bowl_pad, bowl_runs, bowl_mask,
             venue_ids, cat, y, is_inn1) = batch
            team_pred, bat_pred, bowl_pred = model(
                bat_ids, bat_pad, bowl_ids, bowl_pad, venue_ids, cat
            )
            team_loss, n_inn1 = _masked_l1(team_pred, y, is_inn1)
            bat_loss, _ = _masked_l1(bat_pred, bat_runs, bat_mask)
            bowl_loss, _ = _masked_l1(bowl_pred, bowl_runs, bowl_mask)
            loss = team_loss + bat_weight * bat_loss + bowl_weight * bowl_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if n_inn1 > 0:
                tot_team += team_loss.item() * n_inn1
                cnt_team += n_inn1

        train_team_mae = tot_team / max(cnt_team, 1.0)
        val_losses = _eval_losses(model, val_loader)
        train_hist.append(train_team_mae)
        val_hist.append(val_losses["team"])
        val_bat_hist.append(val_losses["bat"])
        val_bowl_hist.append(val_losses["bowl"])
        pbar.set_postfix(
            train=f"{train_team_mae:.1f}",
            val=f"{val_losses['team']:.1f}",
            bat=f"{val_losses['bat']:.1f}",
            bowl=f"{val_losses['bowl']:.1f}",
        )

        if val_losses["team"] < best_val:
            best_val = val_losses["team"]
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
                      f"best val team MAE {best_val:.2f} @ epoch {best_epoch + 1})")
                break

    if not stopped_early:
        print(f"  ran full {epochs} epochs  "
              f"(best val team MAE {best_val:.2f} @ epoch {best_epoch + 1})")

    if best_state is not None:
        model.load_state_dict(best_state)

    if len(val_hist) >= 2:
        print(f"  train team {_sparkline(train_hist)}  [{train_hist[0]:.1f} → {train_hist[-1]:.1f}]")
        print(f"  val   team {_sparkline(val_hist)}  [{val_hist[0]:.1f} → {val_hist[-1]:.1f}]")
        print(f"  val   bat  {_sparkline(val_bat_hist)}  [{val_bat_hist[0]:.1f} → {val_bat_hist[-1]:.1f}]")
        print(f"  val   bowl {_sparkline(val_bowl_hist)}  [{val_bowl_hist[0]:.1f} → {val_bowl_hist[-1]:.1f}]")

    with torch.no_grad():
        player_emb = model.player_embed.weight.detach().cpu().numpy()
        venue_emb = model.venue_embed.weight.detach().cpu().numpy()
    history = {
        "train": train_hist,
        "val": val_hist,
        "val_bat": val_bat_hist,
        "val_bowl": val_bowl_hist,
        "best_epoch": best_epoch,
    }
    return player_emb, venue_emb, history


POOL_AGGS = ("mean", "std", "max")


def pool_by_id(
    pipe_series: pd.Series, vocab: PlayerVocab, matrix: np.ndarray
) -> dict[str, np.ndarray]:
    """Mean / std / max of `matrix[ids]` per match.

    Each value is shape (n_matches, matrix.shape[1]). Mean alone collapses
    per-player signal (one strong batter hidden among ten weak ones looks
    the same as eleven average ones); std + max preserve that tail.
    """
    n, d = len(pipe_series), matrix.shape[1]
    out = {k: np.zeros((n, d), dtype=np.float32) for k in POOL_AGGS}
    for i, s in enumerate(pipe_series):
        ids = vocab.encode(s)
        if ids:
            vals = matrix[ids]
            out["mean"][i] = vals.mean(axis=0)
            out["std"][i] = vals.std(axis=0)
            out["max"][i] = vals.max(axis=0)
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
    rolling_form: pd.DataFrame | None = None,
) -> Dataset:
    """Build Stage-2 features. Caller must filter df to innings==1 first.

    `rolling_form` (if provided) is the output of `build_rolling_form(df)`
    and supplies pre-match recent-form features (team / venue rolling means).
    """
    parts: list[pd.DataFrame] = [
        df[REMAINING_CATS].reset_index(drop=True),
        df[NUMERIC_FEATS].reset_index(drop=True),
    ]

    if rolling_form is not None:
        parts.append(rolling_form.reset_index(drop=True))

    for prefix, col in PLAYER_LIST_COLS.items():
        pooled_emb = pool_by_id(df[col], player_vocab, player_emb)
        for agg, arr in pooled_emb.items():
            parts.append(pd.DataFrame(
                arr,
                columns=[f"{prefix}_emb_{agg}_{i}" for i in range(arr.shape[1])],
            ))
        pooled_stat = pool_by_id(df[col], player_vocab, player_stats_raw)
        for agg, arr in pooled_stat.items():
            parts.append(pd.DataFrame(
                arr,
                columns=[f"{prefix}_stat_{agg}_{c}" for c in PLAYER_STAT_COLS],
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
    """Stratify `frac` of each season's train matches into validation.

    Splits at match_id level so both innings of the same match stay in the
    same split (avoids leaking venue / opponent / players across train/val).
    """
    rng = np.random.default_rng(seed)
    train_rows = df[train_full_mask]
    val_match_ids: set = set()
    for _, grp in train_rows.drop_duplicates("match_id").groupby("season", observed=True):
        n = max(1, int(round(len(grp) * frac)))
        picks = rng.choice(grp["match_id"].to_numpy(), size=n, replace=False)
        val_match_ids.update(picks.tolist())
    val_mask = df["match_id"].isin(val_match_ids) & train_full_mask
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
    verbose: bool = True,
) -> lgb.Booster:
    params = {**LGBM_PARAMS, **(params or {})}
    train_set = lgb.Dataset(X_train, y_train, categorical_feature=categorical_cols)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=categorical_cols)
    callbacks = [lgb.early_stopping(early_stopping_rounds, first_metric_only=True)]
    if verbose:
        callbacks.append(lgb.log_evaluation(50))
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )


@dataclass
class Ensemble:
    """Average of N LGBM boosters trained with different RNG seeds.

    Diversity comes from bagging_seed / feature_fraction_seed only: same
    data, same hyperparameters, different stochastic column and row samples
    per boosting round. Typical squeeze in tabular regression is 0.3-1 MAE.
    """
    boosters: list[lgb.Booster]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.stack(
            [b.predict(X, num_iteration=b.best_iteration) for b in self.boosters]
        )
        return preds.mean(axis=0)


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_cols: list[str],
    n_models: int = 5,
    base_seed: int = VAL_SEED,
    params: dict | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
) -> Ensemble:
    boosters: list[lgb.Booster] = []
    for i in range(n_models):
        seed = base_seed + 101 * i
        variant_params = {
            **(params or {}),
            "seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
        }
        print(f"  ensemble model {i + 1}/{n_models}  seed={seed}")
        b = train_model(
            X_train, y_train, X_val, y_val,
            categorical_cols=categorical_cols,
            params=variant_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        print(f"    best_iter={b.best_iteration}  val_l1={b.best_score['val']['l1']:.3f}")
        boosters.append(b)
    return Ensemble(boosters=boosters)


def evaluate_ensemble(
    ens: Ensemble,
    X: pd.DataFrame,
    y: pd.Series,
    offset: np.ndarray | None = None,
) -> dict[str, float]:
    pred = ens.predict(X)
    if offset is not None:
        pred = pred + offset
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
    }


def evaluate_ensemble_by_season(
    ens: Ensemble,
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    offset: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    pred = ens.predict(X)
    if offset is not None:
        pred = pred + offset
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


def evaluate(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    offset: np.ndarray | None = None,
) -> dict[str, float]:
    pred = model.predict(X, num_iteration=model.best_iteration)
    if offset is not None:
        pred = pred + offset
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
    }


def evaluate_by_season(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    offset: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    pred = model.predict(X, num_iteration=model.best_iteration)
    if offset is not None:
        pred = pred + offset
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


# -------------------------------------------------------- Season-baseline detrending


@dataclass
class SeasonBaseline:
    """Per-season mean team_runs, with weighted-recency extrapolation for
    unseen future years.

    Trees can't extrapolate target values beyond training-leaf means —
    subtracting a level lets predictions naturally shift up for unseen late
    years once the level is re-added at inference. The naive linear fit over
    all train seasons badly undershoots 2024/2025 (the post-2022 "Impact
    Player" scoring jump is not linear), so:

      - train years: use their own in-sample season mean (cleanest residual)
      - future years: extrapolate with recency-weighted linear fit (τ small
        so last 3-5 seasons dominate)
    """
    year_means: dict[int, float]
    slope: float
    intercept: float
    max_train_year: int

    def predict(self, years) -> np.ndarray:
        arr = np.asarray(years, dtype=np.int64)
        out = np.empty(len(arr), dtype=np.float32)
        last_mean = self.year_means[self.max_train_year]
        for i, y in enumerate(arr):
            iy = int(y)
            if iy in self.year_means:
                out[i] = self.year_means[iy]
            elif iy > self.max_train_year:
                # Anchor future extrapolation to last observed season mean;
                # the raw weighted-regression line can sit below the last
                # known point because it's fit to the cluster centroid, not
                # anchored to the endpoint.
                out[i] = last_mean + self.slope * (iy - self.max_train_year)
            else:
                out[i] = self.slope * float(y) + self.intercept
        return out


def fit_season_baseline(
    df_inn1_train: pd.DataFrame, tau: float = SEASON_TREND_TAU
) -> SeasonBaseline:
    """Exponentially-weighted linear extrapolation over per-season means.

    Weights ∝ exp((year - max_year) / τ) so recent years dominate. τ=2 means
    the last ~4 seasons carry >70% of the fit weight — mirrors how T20
    scoring has accelerated since 2022 and avoids the flat pre-2020 era
    dragging the slope down.
    """
    g = df_inn1_train.groupby("year")[TARGET].mean().reset_index()
    years = g["year"].to_numpy(dtype=np.float64)
    vals = g[TARGET].to_numpy(dtype=np.float64)
    w = np.exp((years - years.max()) / tau)
    w /= w.sum()
    wm_y = (w * years).sum()
    wm_v = (w * vals).sum()
    num = (w * (years - wm_y) * (vals - wm_v)).sum()
    den = (w * (years - wm_y) ** 2).sum()
    slope = float(num / den)
    intercept = float(wm_v - slope * wm_y)
    year_means = {int(y): float(v) for y, v in zip(years, vals)}
    return SeasonBaseline(
        year_means=year_means,
        slope=slope,
        intercept=intercept,
        max_train_year=int(years.max()),
    )


def trend_baseline_metrics(
    trend: SeasonBaseline, df_inn1_test: pd.DataFrame
) -> dict[str, float]:
    pred = trend.predict(df_inn1_test["year"].to_numpy())
    y = df_inn1_test[TARGET].to_numpy()
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
    }


# Backward-compat alias in case external code imports the old name.
YearTrend = SeasonBaseline
fit_year_trend = fit_season_baseline


# ------------------------------------------------------ Rolling-form features


def build_rolling_form(
    df_inn1: pd.DataFrame,
    bat_windows: tuple[int, ...] = BAT_ROLL_WINDOWS,
    bowl_windows: tuple[int, ...] = BOWL_ROLL_WINDOWS,
    venue_windows: tuple[int, ...] = VENUE_ROLL_WINDOWS,
) -> pd.DataFrame:
    """Per-match rolling form features from strictly-prior innings-1 rows.

    Each feature is a trailing mean / std of team_runs grouped by
    batting_team, bowling_team, or venue, shifted by 1 so the current match
    never leaks into its own history. Also emits season-to-date expanding
    means (stabler than short rolling means early in a season).

    Sorting is by (date, match_id) — the only globally-chronological ordering
    we have (raw match_id is ingestion order, not date order).

    Returns a DataFrame aligned 1:1 with the input (same index).
    """
    work = df_inn1.copy()
    work["_order"] = np.arange(len(work))
    work = work.sort_values(["date", "match_id"], kind="stable")

    feats = pd.DataFrame(index=work.index)

    def _rolling_mean(group_col: str, windows, prefix: str) -> None:
        for K in windows:
            feats[f"{prefix}_last{K}"] = (
                work.groupby(group_col, observed=True)[TARGET]
                .transform(lambda s: s.shift(1).rolling(K, min_periods=1).mean())
                .astype(np.float32)
            )

    def _rolling_std(group_col: str, window: int, prefix: str) -> None:
        """Volatility: large rolling std means team/venue is inconsistent."""
        feats[f"{prefix}_std{window}"] = (
            work.groupby(group_col, observed=True)[TARGET]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
            .astype(np.float32)
        )

    def _season_to_date(group_col: str, prefix: str) -> None:
        """Expanding mean reset each season — stable baseline even when a
        team has played few matches so far this year."""
        feats[f"{prefix}_sea"] = (
            work.groupby(["season", group_col], observed=True)[TARGET]
            .transform(lambda s: s.shift(1).expanding().mean())
            .astype(np.float32)
        )

    _rolling_mean("batting_team", bat_windows, "bat_form")
    _rolling_mean("bowling_team", bowl_windows, "bowl_form")
    _rolling_mean("venue", venue_windows, "venue_form")

    _rolling_std("batting_team", 10, "bat_form")
    _rolling_std("bowling_team", 10, "bowl_form")
    _rolling_std("venue", 20, "venue_form")

    _season_to_date("batting_team", "bat_form")
    _season_to_date("bowling_team", "bowl_form")
    _season_to_date("venue", "venue_form")

    # Cold-start NaN: first match for a team / venue has no prior history.
    # Leaving as NaN lets LGBM route them down a dedicated branch; better
    # than injecting an arbitrary mean that could bias learning.

    # Restore original row order.
    feats["_order"] = work["_order"].to_numpy()
    feats = feats.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return feats

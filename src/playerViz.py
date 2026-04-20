"""Interactive IPL player embedding explorer.

Clusters the Stage 1 player embeddings and visualises them in a tkinter window.
Projection: PCA or t-SNE. Clustering: KMeans.

Run `python main.py` first to generate `archive/player_embeddings.npz`,
then run `python playerViz.py`.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ARCHIVE = Path(__file__).resolve().parent.parent / "archive"
STAT_COLS = ["runs", "strike_rate", "wickets", "balls_bowled", "economy"]

_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
]


class PlayerEmbeddingExplorer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("IPL Player Embedding Explorer")
        root.geometry("1280x780")

        self._load_data()

        self.n_clusters_var = tk.IntVar(value=6)
        self.method_var = tk.StringVar(value="t-SNE")
        self.cluster_filter_var = tk.StringVar(value="All")
        self.coords_2d: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self._list_indices: list[int] = []

        self._build_ui()
        self._recompute()

    # ------------------------------------------------------------------ data

    def _load_data(self) -> None:
        emb_path = ARCHIVE / "player_embeddings.npz"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"{emb_path} not found.\nRun `python main.py` first to generate it."
            )
        data = np.load(emb_path, allow_pickle=True)
        all_emb: np.ndarray = data["embeddings"]   # (vocab_size, embed_dim)
        all_names: np.ndarray = data["names"]       # (vocab_size,) object

        stats_df = pd.read_csv(ARCHIVE / "player_stats.csv").set_index("player")

        valid_idx, names, stats = [], [], []
        for i in range(1, len(all_names)):          # skip PAD at index 0
            name = str(all_names[i])
            valid_idx.append(i)
            names.append(name)
            if name in stats_df.index:
                row = stats_df.loc[name]
                stats.append({
                    "runs": int(row["runs"]),
                    "strike_rate": round(float(row["strike_rate"]), 1),
                    "wickets": int(row["wickets"]),
                    "balls_bowled": int(row["balls_bowled"]),
                    "economy": round(float(row["economy"]), 2),
                })
            else:
                stats.append({"runs": 0, "strike_rate": 0.0,
                               "wickets": 0, "balls_bowled": 0, "economy": 0.0})

        self.embeddings = all_emb[valid_idx]
        self.names = names
        self.stats = stats

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        # Status bar
        self.status_var = tk.StringVar(value="Hover over a point · click to select")
        tk.Label(
            self.root, textvariable=self.status_var, anchor="w",
            relief=tk.SUNKEN, bg="#e0e0e0", font=("Arial", 9),
        ).pack(side=tk.BOTTOM, fill=tk.X, ipady=3)

        # Main pane: plot left, controls right
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5, bg="#ccc")
        pane.pack(fill=tk.BOTH, expand=True)

        # ── left: matplotlib ────────────────────────────────────────
        plot_frame = tk.Frame(pane)
        pane.add(plot_frame, minsize=700)

        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame).update()

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        # ── right: controls ─────────────────────────────────────────
        right = tk.Frame(pane, bg="#f4f4f4", padx=10, pady=10)
        pane.add(right, minsize=270)

        tk.Label(right, text="Player Embedding Explorer",
                 font=("Arial", 11, "bold"), bg="#f4f4f4").pack(pady=(0, 10))

        # Clusters
        row = tk.Frame(right, bg="#f4f4f4")
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="Clusters:", bg="#f4f4f4", width=11, anchor="w").pack(side=tk.LEFT)
        tk.Spinbox(row, from_=2, to=15, textvariable=self.n_clusters_var,
                   width=4, command=self._recompute).pack(side=tk.LEFT)

        # Projection
        tk.Label(right, text="Projection:", bg="#f4f4f4", anchor="w").pack(fill=tk.X, pady=(8, 2))
        for m in ("PCA", "t-SNE"):
            tk.Radiobutton(right, text=m, variable=self.method_var, value=m,
                           bg="#f4f4f4", command=self._recompute).pack(anchor="w", padx=14)

        tk.Button(right, text="⟳  Recompute", command=self._recompute,
                  relief=tk.GROOVE, pady=4).pack(fill=tk.X, pady=8)

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=4)

        # Cluster filter
        tk.Label(right, text="Filter by cluster:", bg="#f4f4f4", anchor="w").pack(fill=tk.X)
        self.cluster_combo = ttk.Combobox(
            right, textvariable=self.cluster_filter_var, state="readonly")
        self.cluster_combo.pack(fill=tk.X, pady=(2, 6))
        self.cluster_combo.bind("<<ComboboxSelected>>",
                                lambda _: self._update_player_list())

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=4)

        # Player list
        tk.Label(right, text="Players  (runs · wickets):",
                 bg="#f4f4f4", anchor="w").pack(fill=tk.X)
        list_frame = tk.Frame(right)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        sb = tk.Scrollbar(list_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.player_list = tk.Listbox(
            list_frame, yscrollcommand=sb.set,
            font=("Courier", 9), activestyle="dotbox",
            selectbackground="#4a90d9", selectforeground="white",
        )
        self.player_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.player_list.yview)
        self.player_list.bind("<<ListboxSelect>>", self._on_list_select)

    # ------------------------------------------------------------------ compute

    def _recompute(self) -> None:
        n = self.n_clusters_var.get()

        self.labels = KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(
            self.embeddings
        )

        if self.method_var.get() == "t-SNE":
            perp = min(30, len(self.embeddings) - 1)
            self.coords_2d = TSNE(
                n_components=2, perplexity=perp, random_state=42,
            ).fit_transform(self.embeddings)
        else:
            self.coords_2d = PCA(n_components=2).fit_transform(self.embeddings)

        options = ["All"] + [f"Cluster {i}" for i in range(n)]
        self.cluster_combo["values"] = options
        self.cluster_filter_var.set("All")

        self._draw_scatter()
        self._update_player_list()

    # ------------------------------------------------------------------ drawing

    def _color(self, cluster_id: int) -> str:
        return _COLORS[cluster_id % len(_COLORS)]

    def _draw_scatter(self, highlight: int | None = None) -> None:
        self.ax.clear()
        n = self.n_clusters_var.get()

        for cid in range(n):
            mask = self.labels == cid
            self.ax.scatter(
                self.coords_2d[mask, 0], self.coords_2d[mask, 1],
                color=self._color(cid), s=22, alpha=0.72, label=f"C{cid}",
                linewidths=0,
            )

        if highlight is not None:
            x, y = self.coords_2d[highlight]
            self.ax.scatter(x, y, s=160, color="gold", edgecolors="black",
                            linewidths=1.2, zorder=6, marker="*")
            self.ax.annotate(
                self.names[highlight], (x, y),
                xytext=(7, 7), textcoords="offset points",
                fontsize=8, zorder=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc",
                          ec="#aaa", alpha=0.9),
            )

        method = self.method_var.get()
        self.ax.set_title(
            f"{method}  ·  {n} clusters  ·  {len(self.names)} players",
            fontsize=11,
        )
        self.ax.set_xlabel("component 1")
        self.ax.set_ylabel("component 2")
        ncol = 2 if n > 8 else 1
        self.ax.legend(fontsize=7, markerscale=2.0, loc="best",
                       framealpha=0.7, ncol=ncol)
        self.ax.grid(True, alpha=0.2)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ list

    def _update_player_list(self) -> None:
        self.player_list.delete(0, tk.END)
        val = self.cluster_filter_var.get()
        if val == "All":
            indices = list(range(len(self.names)))
        else:
            cid = int(val.split()[-1])
            indices = [i for i, lbl in enumerate(self.labels) if lbl == cid]

        # Sort: primarily by runs desc, secondarily by wickets desc
        indices.sort(key=lambda i: (-self.stats[i]["runs"], -self.stats[i]["wickets"]))
        self._list_indices = indices

        for i in indices:
            name = self.names[i]
            r = self.stats[i]["runs"]
            w = self.stats[i]["wickets"]
            self.player_list.insert(tk.END, f"{name[:25]:<25}  {r:>5}r  {w:>3}w")

    def _scroll_to(self, list_pos: int) -> None:
        self.player_list.see(list_pos)
        self.player_list.selection_clear(0, tk.END)
        self.player_list.selection_set(list_pos)

    # ------------------------------------------------------------------ events

    def _nearest(self, event) -> int | None:
        """Nearest point to the cursor in pixel space (robust to axis scaling)."""
        if self.coords_2d is None or event.x is None or event.y is None:
            return None
        pix = self.ax.transData.transform(self.coords_2d)
        dx = pix[:, 0] - event.x
        dy = pix[:, 1] - event.y
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))
        return idx if d2[idx] < 18 * 18 else None  # within 18px

    def _status(self, idx: int) -> None:
        s = self.stats[idx]
        cid = self.labels[idx]
        self.status_var.set(
            f"{self.names[idx]}  │  cluster {cid}  │  "
            f"runs={s['runs']}  SR={s['strike_rate']}  "
            f"wkts={s['wickets']}  econ={s['economy']}"
        )

    def _on_hover(self, event) -> None:
        if event.inaxes != self.ax or self.coords_2d is None:
            return
        idx = self._nearest(event)
        if idx is None:
            self.status_var.set("Hover over a point · click to select")
        else:
            self._status(idx)

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or self.coords_2d is None:
            return
        idx = self._nearest(event)
        if idx is None:
            return
        cid = self.labels[idx]
        # Switch filter to this cluster and highlight the player
        self.cluster_filter_var.set(f"Cluster {cid}")
        self._update_player_list()
        self._draw_scatter(highlight=idx)
        try:
            pos = self._list_indices.index(idx)
            self._scroll_to(pos)
        except ValueError:
            pass
        self._status(idx)

    def _on_list_select(self, event) -> None:
        sel = self.player_list.curselection()
        if not sel:
            return
        idx = self._list_indices[sel[0]]
        self._draw_scatter(highlight=idx)
        self._status(idx)


def main() -> None:
    root = tk.Tk()
    try:
        PlayerEmbeddingExplorer(root)
    except FileNotFoundError as exc:
        tk.messagebox.showerror("Missing data", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()

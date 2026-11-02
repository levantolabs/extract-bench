"""Draw the README charts from results/<version>/board.json (run `extract-bench report` first).

  pip install -e ".[figures]"
  python scripts/make_figures.py            # -> docs/images/{leaderboard,per-dataset,sage2-vs-clairic}{,-dark}.png

Colours: one blue for every system (no system gets its own brand colour), a single-hue blue ramp for the heatmap,
neutral greys for axes and grid. Light and dark variants use the same hues stepped for each surface.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v0.1"
OUT = ROOT / "docs" / "images"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984", "grid": "#e6e5e1",
              "zero": "#b9b8b2", "mark": "#2a78d6", "track": "#efeeea", "ramp": RAMP},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#8f8e86", "grid": "#2f2f2c",
             "zero": "#5b5a55", "mark": "#3987e5", "track": "#262624", "ramp": RAMP[::-1]},
}
NAMES = {"fatura": "FATURA", "omniai": "OmniAI", "docvqa": "DocVQA", "vrdu": "VRDU", "realdoc": "RealDoc",
         "kleister_nda": "Kleister-NDA", "kleister_charity": "Kleister-Charity", "emails": "Emails", "tatqa": "TAT-QA",
         "squad2": "SQuAD 2.0", "sroie": "SROIE", "cord": "CORD"}
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})


def _frame(ax, T):
    ax.set_facecolor(T["surface"])
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["grid"])
    ax.tick_params(colors=T["text2"], length=0)
    ax.xaxis.label.set_color(T["text2"])


def leaderboard(b, T, path):
    B = b["boards"]["text"]; S = b["systems"]
    order = sorted(B["systems"], key=lambda s: B["mean"][s])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08},
                             facecolor=T["surface"])
    panels = [("Field accuracy", B["mean"]), ('"Not in the document" answered correctly', B["absent_correct_mean"])]
    for k, (ax, (title, vals)) in enumerate(zip(axes, panels)):
        _frame(ax, T)
        y = range(len(order))
        ax.barh(y, [100] * len(order), height=0.52, color=T["track"], zorder=1)
        ax.barh(y, [vals[s] for s in order], height=0.52, color=T["mark"], zorder=2)
        for i, s in enumerate(order):
            ax.text(vals[s] + 1.5, i, f"{vals[s]:.1f}", va="center", ha="left", color=T["text"], fontsize=10,
                    fontweight="bold", zorder=3)
        ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.grid(True, color=T["grid"], linewidth=0.8, zorder=0); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=T["text"], fontsize=11, fontweight="bold", pad=8)
        ax.spines["left"].set_visible(False)
        if k == 0:
            ax.set_yticks(list(y))
            ax.set_yticklabels([f"{S[s]['display_name']}{' *' if S[s].get('maintainer_product') else ''}" for s in order],
                               color=T["text"], fontsize=10.5)
            for i, s in enumerate(order):
                ax.text(-0.02, i - 0.36, f"{S[s]['org']} · {'open weights' if S[s]['open_weights'] else 'API'}",
                        transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, color=T["muted"])
        else:
            ax.set_yticks([])
    fig.text(0.01, 0.015, f"Extract Bench {b['version']} · mean of {len(B['mean_sets'])} datasets · held-out half · "
             "* maintainer's product (Levanto)", color=T["muted"], fontsize=8)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.14)
    fig.savefig(path, facecolor=T["surface"]); plt.close(fig)


def per_dataset(b, T, path):
    B = b["boards"]["text"]; S = b["systems"]
    order = sorted(B["systems"], key=lambda s: -B["mean"][s])
    ds = list(B["datasets"])
    grid = [[B["datasets"][d]["acc"][s] for d in ds] for s in order]
    cmap = LinearSegmentedColormap.from_list("seq", T["ramp"])
    fig, ax = plt.subplots(figsize=(11, 3.3), facecolor=T["surface"]); _frame(ax, T)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            dark_cell = (v > 55) if T is THEMES["light"] else (v < 45)
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if dark_cell else T["text"])
    ax.set_xticks(range(len(ds))); ax.set_xticklabels([NAMES[d] for d in ds], rotation=30, ha="right", color=T["text"])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([S[s]["display_name"] for s in order], color=T["text"])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    n_main = len(B["mean_sets"])
    ax.axvline(n_main - 0.5, color=T["surface"], linewidth=4)
    ax.text(n_main + 0.5, -0.75, "receipts (not in the mean)", ha="center", va="bottom", fontsize=8, color=T["muted"])
    ax.set_title("Field accuracy per dataset, held-out half", loc="left", color=T["text"], fontsize=11,
                 fontweight="bold", pad=14)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.85, bottom=0.25)
    fig.savefig(path, facecolor=T["surface"]); plt.close(fig)


def sage2_vs_clairic(b, T, path):
    B = b["boards"]["headline"]
    rows = sorted(((d, r["diff_vs_ref"]["clairic"]) for d, r in B["datasets"].items() if d in B["mean_sets"]),
                  key=lambda x: -x[1]["diff"])           # Clairic − Sage 2; flip sign: Sage 2 − Clairic
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=T["surface"]); _frame(ax, T)
    for i, (d, v) in enumerate(rows):
        lo, hi, x = -v["ci95"][1], -v["ci95"][0], -v["diff"]
        ax.plot([lo, hi], [i, i], color=T["mark"], linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([x], [i], s=64, zorder=3, color=T["mark"] if v["sig"] else T["surface"],
                   edgecolors=T["mark"], linewidths=2)
        lab = "±0.0" if abs(x) < 0.05 else f"{x:+.1f}".replace("-", "−")
        ax.text(max(hi, x) + 1.2, i, lab, va="center", fontsize=9, color=T["text"])
    ax.axvline(0, color=T["zero"], linewidth=1, zorder=1)
    ax.set_xlim(min(-v["ci95"][1] for _, v in rows) - 2, max(-v["ci95"][0] for _, v in rows) + 7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([NAMES[d] for d, _ in rows], color=T["text"])
    ax.invert_yaxis(); ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=T["grid"], linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel("← Clairic ahead      Sage 2 − Clairic, accuracy points      Sage 2 ahead →", fontsize=9)
    ax.set_title("Sage 2 vs Clairic per dataset (95% bootstrap CI)", loc="left", color=T["text"], fontsize=11,
                 fontweight="bold", pad=24)
    h1 = ax.scatter([], [], s=64, color=T["mark"], edgecolors=T["mark"], linewidths=2, label="CI excludes 0")
    h2 = ax.scatter([], [], s=64, color=T["surface"], edgecolors=T["mark"], linewidths=2, label="CI includes 0")
    leg = ax.legend(handles=[h1, h2], loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
                    fontsize=8.5, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(T["text2"])
    fig.subplots_adjust(left=0.2, right=0.97, top=0.84, bottom=0.14)
    fig.savefig(path, facecolor=T["surface"]); plt.close(fig)


def main():
    b = json.loads((ROOT / "results" / VERSION / "board.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for name, T in THEMES.items():
        sfx = "" if name == "light" else "-dark"
        leaderboard(b, T, OUT / f"leaderboard{sfx}.png")
        per_dataset(b, T, OUT / f"per-dataset{sfx}.png")
        sage2_vs_clairic(b, T, OUT / f"sage2-vs-clairic{sfx}.png")
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()

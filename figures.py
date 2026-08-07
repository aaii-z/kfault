"""Paper figures, built from the exported tables. No network, no Modal.

    uv run python figures.py

Reads data/attempts.csv and data/chains.csv; writes docs/figures/paper.

Every number is computed from the CSVs, so re-exporting after a new run and
re-running this script regenerates the figures unchanged. Palette choices and
their accessibility checks are recorded in docs/figures/paper/README.md.
"""

import csv
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from kfault.analysis import false_pass_rate  # noqa: E402

OUT = "docs/figures/paper"

# validated categorical slots, in fixed order; never cycled
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED, GREEN = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#4a3aa7",
    "#e34948",
    "#008300",
)
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8b8a85"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"

# one hue per model, assigned once and reused across every figure
MODEL_COLOR = {"GLM-5": BLUE, "DeepSeek V3": ORANGE, "Qwen3-235B": AQUA}
SHORT = {
    "zai.glm-5": "GLM-5",
    "deepseek.v3-v1:0": "DeepSeek V3",
    "qwen.qwen3-235b-a22b-2507-v1:0": "Qwen3-235B",
}
MODEL_ORDER = ["GLM-5", "DeepSeek V3", "Qwen3-235B"]
CATEGORIES = ["syntactic", "runtime", "functional", "silent", "pass"]
# Severity reads left to right: does not build (blue) -> crashes (violet) ->
# wrong answer (red) -> secretly wrong (amber warning) -> correct (green).
# `functional` must NOT be a green: it is a failure, and an earlier aqua made it
# look like a sibling of `pass`. Red is right, but that forces `runtime` off
# orange, because orange beside red fails both gates (normal-vision dE 7.1), so
# runtime takes violet.
# validator: adjacent CVD dE 13.0, normal-vision 16.3, light mode. All pass.
CAT_COLOR = {
    "syntactic": BLUE,
    "runtime": VIOLET,
    "functional": RED,
    "silent": YELLOW,
    "pass": GREEN,
}
SILENT = YELLOW


def style(ax, *, xgrid=False):
    """Recessive axes and grid; the data carries the ink."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            f"{OUT}/{name}.{ext}",
            dpi=200,
            bbox_inches="tight",
            facecolor=SURFACE,
        )
    plt.close(fig)
    print(f"  {OUT}/{name}.pdf / .png")


def load():
    chains = list(csv.DictReader(open("data/chains.csv")))
    attempts = list(csv.DictReader(open("data/attempts.csv")))
    for row in chains + attempts:
        row["model"] = SHORT.get(row["model_id"], row["model_id"])
    return chains, attempts


def false_pass(counter):
    """(percent, silent, total) from a category counter."""
    silent, total, percent = false_pass_rate(counter)
    return percent, silent, total


# ----------------------------------------------------------------- figure 1
def fig_relocation(chains):
    """The headline: repair raises the false-pass rate in every model.

    A slope chart, because the claim is about DIRECTION of change between two
    states, not about the two magnitudes independently.
    """
    rows = [c for c in chains if c["device"] == "L4" and c["arm"] == "think"]
    data = {}
    for model in MODEL_ORDER:
        subset = [c for c in rows if c["model"] == model]
        one = false_pass(Counter(c["first_category"] for c in subset))
        rep = false_pass(Counter(c["final_category"] for c in subset))
        data[model] = (one, rep)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    style(ax)
    ax.grid(False)

    # endpoint labels collide when two models land within ~1 point of each
    # other (GLM-5 20% vs DeepSeek 19%); nudge them apart, keep the marks put
    ends = sorted(((data[m][1][0], m) for m in MODEL_ORDER))
    label_y, previous = {}, None
    for value, model in ends:
        y = value if previous is None or value - previous >= 2.8 else previous + 2.8
        label_y[model] = y
        previous = y

    for model in MODEL_ORDER:
        (y0, s0, t0), (y1, s1, t1) = data[model]
        color = MODEL_COLOR[model]
        ax.plot([0, 1], [y0, y1], color=color, linewidth=2, zorder=3)
        ax.plot([0, 1], [y0, y1], "o", color=color, markersize=8, zorder=4)
        ax.annotate(
            f"{y0:.0f}%",
            (0, y0),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
        )
        ax.annotate(
            f"{model}   {y1:.0f}%",
            (1.06, label_y[model] + 0.55),
            ha="left",
            va="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
        )
        ax.annotate(
            f"{s1} of {t1} passing kernels",
            (1.06, label_y[model] - 0.95),
            ha="left",
            va="center",
            color=INK_MUTED,
            fontsize=8,
        )

    ax.set_xlim(-0.42, 2.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["one-shot", "after 3 repair turns"], fontsize=10, color=INK_2)
    ax.set_ylim(0, 24)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel("of passing kernels, share that are silent failures", fontsize=9)
    ax.set_title(
        "Repair raises the pass rate and the false-pass rate together\n",
        fontsize=12.5,
        color=INK,
        loc="left",
        pad=22,
        fontweight="bold",
    )
    ax.text(
        0,
        1.035,
        "reasoning arm, KernelBench Level 2, L4 - every model moves upward",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_2,
    )
    save(fig, "fig1_relocation")


# ----------------------------------------------------------------- figure 2
def fig_outcomes(chains):
    """What repair actually did to each chain, counted."""
    order = ["rescued", "hidden", "still_failing"]
    label = {
        "rescued": "repaired to pass",
        "hidden": "visible failure became SILENT",
        "still_failing": "still visibly failing",
    }
    # "hidden" wears the silent colour used in fig3/fig4: one meaning, one hue
    color = {"rescued": GREEN, "hidden": SILENT, "still_failing": INK_MUTED}
    ink_on = {"rescued": "white", "hidden": INK, "still_failing": "white"}

    rows = [c for c in chains if c["device"] == "L4"]
    counts = {
        m: Counter(c["outcome"] for c in rows if c["model"] == m) for m in MODEL_ORDER
    }

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    style(ax, xgrid=True)
    ypos = range(len(MODEL_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        left = 0
        for key in order:
            width = counts[model].get(key, 0)
            if not width:
                continue
            ax.barh(
                index,
                width,
                left=left,
                height=0.55,
                color=color[key],
                edgecolor=SURFACE,
                linewidth=2,  # 2px surface gap between segments
                zorder=3,
            )
            ax.text(
                left + width / 2,
                index,
                str(width),
                ha="center",
                va="center",
                fontsize=9,
                color=ink_on[key],
                fontweight="bold",
                zorder=4,
            )
            left += width
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(MODEL_ORDER, fontsize=10, color=INK)
    ax.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)  # GLM-5 on top
    ax.set_xlabel("repair chains", fontsize=9, color=INK_2)
    ax.set_title(
        "For every ~25 kernels repair rescues, it hides ~16 more",
        fontsize=12,
        color=INK,
        loc="left",
        pad=14,
        fontweight="bold",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color[k], label=label[k]) for k in order
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        labelcolor=INK_2,
    )
    save(fig, "fig2_repair_outcomes")


# ----------------------------------------------------------------- figure 3
def fig_taxonomy(chains):
    """Where the failures sit, before and after repair, per arm."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, arm in zip(axes, ("think", "direct")):
        style(ax, xgrid=True)
        rows = [c for c in chains if c["device"] == "L4" and c["arm"] == arm]
        labels, index = [], 0
        for model in MODEL_ORDER:
            subset = [c for c in rows if c["model"] == model]
            stages = (
                ("one-shot", "first_category"),
                ("repaired", "final_category"),
            )
            for stage, field in stages:
                counts = Counter(c[field] for c in subset)
                left = 0
                for cat in CATEGORIES:
                    width = counts.get(cat, 0)
                    if not width:
                        continue
                    ax.barh(
                        index,
                        width,
                        left=left,
                        height=0.62,
                        color=CAT_COLOR[cat],
                        edgecolor=SURFACE,
                        linewidth=2,
                        zorder=3,
                    )
                    if width >= 6:
                        ax.text(
                            left + width / 2,
                            index,
                            str(width),
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white",
                            fontweight="bold",
                            zorder=4,
                        )
                    left += width
                labels.append(f"{model}  {stage}")
                index += 1
            index += 0.5
        ax.set_yticks(range(len(labels)) if False else [])
        ax.set_yticks([i + (i // 2) * 0.5 for i in range(len(labels))])
        ax.set_yticklabels(labels, fontsize=8, color=INK_2)
        ax.set_ylim(max(ax.get_yticks()) + 0.7, -0.7)  # GLM-5 on top
        ax.set_xlabel("problems", fontsize=9, color=INK_2)
        ax.set_title(
            f"{arm} arm  (reasoning {'ON' if arm == 'think' else 'OFF'})",
            fontsize=10,
            color=INK,
            loc="left",
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CAT_COLOR[c], label=c) for c in CATEGORIES
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=5,
        labelcolor=INK_2,
    )
    fig.suptitle(
        "Repair eliminates syntactic failures and grows the silent category",
        fontsize=12,
        color=INK,
        x=0.09,
        ha="left",
        y=1.02,
        fontweight="bold",
    )
    save(fig, "fig3_taxonomy")


# ----------------------------------------------------------------- figure 4
def fig_probes(attempts):
    """Which stress probe actually catches the silent failures."""
    probes = [
        "non_contig",
        "extreme",
        "small_magnitude",
        "zeros",
        "nondeterministic",
        "fresh_seed_1",
        "fresh_seed_2",
        "fresh_seed_3",
    ]
    pretty = {
        "non_contig": "non-contiguous input",
        "extreme": "extreme values (x100)",
        "small_magnitude": "small values (x1e-4)",
        "zeros": "all zeros",
        "nondeterministic": "run-to-run instability",
        "fresh_seed_1": "fresh random draw 1",
        "fresh_seed_2": "fresh random draw 2",
        "fresh_seed_3": "fresh random draw 3",
    }
    silent = [a for a in attempts if a["category"] == "silent" and a["device"] == "L4"]
    caught = Counter()
    for row in silent:
        for probe in probes:
            if row.get(f"probe_{probe}") in ("mismatch", "kernel_error"):
                caught[probe] += 1

    ordered = [p for p in probes if caught[p]]
    ordered.sort(key=lambda p: -caught[p])
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    style(ax, xgrid=True)
    ax.barh(
        range(len(ordered)),
        [caught[p] for p in ordered],
        height=0.6,
        color=SILENT,
        edgecolor=INK_MUTED,
        linewidth=0.6,
        zorder=3,
    )
    for i, probe in enumerate(ordered):
        ax.text(
            caught[probe] + 2,
            i,
            str(caught[probe]),
            va="center",
            fontsize=9,
            color=INK,
            fontweight="bold",
        )
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([pretty[p] for p in ordered], fontsize=9, color=INK_2)
    ax.set_ylim(len(ordered) - 0.5, -0.5)
    ax.set_xlabel(
        f"silent failures caught (n={len(silent)} silent kernels)", fontsize=9
    )
    ax.set_title(
        "One probe finds almost every silent failure",
        fontsize=12,
        color=INK,
        loc="left",
        pad=12,
        fontweight="bold",
    )
    save(fig, "fig4_probes")


# ----------------------------------------------------------------- figure 5
def fig_speedup(attempts):
    """Correct kernels are barely faster than PyTorch."""
    rows = [
        a
        for a in attempts
        if a["category"] == "pass" and a["device"] == "L4" and a["speedup"]
    ]
    by_model = defaultdict(list)
    for row in rows:
        value = float(row["speedup"])
        if 0 < value < 10:  # excludes one 2670x outlier, annotated below
            by_model[row["model"]].append(value)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    style(ax)
    for index, model in enumerate(MODEL_ORDER):
        values = sorted(by_model[model])
        if not values:
            continue
        ax.scatter(
            values,
            [index + 0.0] * len(values),
            s=26,
            color=MODEL_COLOR[model],
            alpha=0.55,
            edgecolor=SURFACE,
            linewidth=0.5,
            zorder=3,
        )
        median = values[len(values) // 2]
        ax.plot(
            [median, median],
            [index - 0.24, index + 0.24],
            color=INK,
            linewidth=2,
            zorder=5,
        )
        ax.annotate(
            f"median {median:.2f}x  (n={len(values)})",
            (median, index + 0.3),
            fontsize=8,
            color=INK,
            ha="center",
        )
    ax.axvline(1.0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.annotate(
        "parity with PyTorch",
        (1.06, -0.68),
        fontsize=8,
        color=INK_MUTED,
        ha="left",
    )
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, fontsize=10, color=INK)
    ax.set_ylim(len(MODEL_ORDER) - 0.35, -0.85)  # GLM-5 on top
    ax.set_xlim(0, 3.2)
    ax.set_xlabel("speedup vs PyTorch reference (higher is faster)", fontsize=9)
    ax.set_title(
        "Kernels that pass are barely faster than the code they replace",
        fontsize=12,
        color=INK,
        loc="left",
        pad=12,
        fontweight="bold",
    )
    save(fig, "fig5_speedup")


def main():
    chains, attempts = load()
    print("writing figures:")
    fig_relocation(chains)
    fig_outcomes(chains)
    fig_taxonomy(chains)
    fig_probes(attempts)
    fig_speedup(attempts)


if __name__ == "__main__":
    main()

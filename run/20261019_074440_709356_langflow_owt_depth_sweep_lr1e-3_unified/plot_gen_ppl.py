"""Gen PPL vs student depth for the lr=1e-3 depth sweep (teacher = langflow-owt, 12L)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent
SRC = RUN / "20261019_180432_955738_unified_gen_ppl" / "summary.csv"

# Validated default palette (references/palette.md), light mode.
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"
STUDENT = "#2a78d6"   # categorical slot 1 (blue)
TEACHER = "#eb6834"   # categorical slot 2 (orange)
GRID = "#e6e5e2"

# CJK-capable font if available, else fall back silently (set before any text is drawn).
for _f in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Source Han Sans SC", "SimHei", "Arial Unicode MS"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

rows = {r["name"]: r for r in csv.DictReader(SRC.open())}
depths = [6, 7, 8, 9, 10, 11]
x = depths
y = [float(rows[f"student_{d:02d}_layers"]["gen_ppl"]) for d in depths]
teacher = float(rows["teacher_12_layers"]["gen_ppl"])

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

# Teacher reference baseline across the plot.
ax.axhline(teacher, color=TEACHER, lw=1.5, ls=(0, (5, 4)), zorder=1)
ax.scatter([12], [teacher], s=70, color=TEACHER, zorder=5, clip_on=False)

# Student line.
ax.plot(x, y, color=STUDENT, lw=2, marker="o", ms=8,
        markerfacecolor=STUDENT, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)

# Direct value labels on students.
for xi, yi in zip(x, y):
    ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                xytext=(0, 11), ha="center", fontsize=9, color=INK)
# Teacher labels.
ax.annotate(f"teacher (12L) {teacher:.1f}", (12, teacher), textcoords="offset points",
            xytext=(-6, 12), ha="right", fontsize=9, color=TEACHER, fontweight="bold")

ax.set_xlim(5.6, 12.5); ax.set_ylim(0, 70)
ax.set_xticks([6, 7, 8, 9, 10, 11, 12])
ax.set_xlabel("Model depth (layers)", fontsize=11, color=INK2)
ax.set_ylabel("Gen PPL  (gpt2-large, lower is better)", fontsize=11, color=INK2)
ax.set_title("Depth-distillation Gen PPL  (lr=1e-3, 5000 steps, 1024 samples)",
             fontsize=13, color=INK, fontweight="bold", pad=14)

ax.grid(True, axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2)

# Legend: two entities, so identity is never color-alone.
from matplotlib.lines import Line2D
legend = [Line2D([0], [0], color=STUDENT, lw=2, marker="o", ms=7,
                 markerfacecolor=STUDENT, markeredgecolor=SURFACE, label="Distilled students"),
          Line2D([0], [0], color=TEACHER, lw=1.5, ls=(0, (5, 4)), marker="o", ms=7,
                 label="Teacher baseline (langflow-owt)")]
ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=9.5)

OUT = RUN / "gen_ppl_depth.png"
fig.tight_layout()
fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
print("wrote", OUT)

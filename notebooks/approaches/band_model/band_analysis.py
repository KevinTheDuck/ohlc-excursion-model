import sys
from pathlib import Path
from datetime import date as dt_date
import warnings
import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJ / "src"))

from ohlc_dss_model.data import load_raw_data
from ohlc_dss_model.data.pipeline_loaders import load_aggregated_data
from ohlc_dss_model.features.band_state import get_band_state_on_ps2

warnings.filterwarnings("ignore")
BURN_IN = dt_date(2011, 4, 5)
OUT = str(_PROJ / "reports")
Path(OUT).mkdir(exist_ok=True)

C = {
    "navy": "#1f3a5f",
    "blue": "#4e79a7",
    "sky": "#86bcb6",
    "teal": "#2a9d8f",
    "green": "#59a14f",
    "lime": "#a0c55f",
    "orange": "#f28e2b",
    "coral": "#e15759",
    "red": "#c44e52",
    "purple": "#b07aa1",
    "violet": "#6c5ce7",
    "pink": "#e8a0bf",
    "gray": "#8c8c8c",
    "charcoal": "#2d3436",
    "white": "#f8f9fa",
}

ZONE_COLORS = [
    C["gray"],
    C["blue"],
    C["coral"],
    C["teal"],
    C["purple"],
    C["green"],
    C["red"],
]

STY = {
    "axes.edgecolor": C["charcoal"],
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.color": C["gray"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
}

print("=" * 60)
print("  BAND STATISTICAL ANALYSIS")
print("=" * 60)

print("\nLoading data...")
raw_30m = load_raw_data(str(_PROJ / "data" / "raw" / "nq_30m.parquet"))
agg = load_aggregated_data(raw_30m)
agg = get_band_state_on_ps2(agg)
agg = agg.filter(pl.col("Session") >= BURN_IN)

sessions = agg["Session"].to_list()
n = agg.height

agg = agg.with_columns(
    [
        (pl.col("Sigma_Historical").shift(1) * pl.col("O_Ref")).alias("sigma_price"),
        (pl.col("Band_AE_Pos_Upper") - pl.col("Band_AE_Neg_Lower")).alias(
            "total_width"
        ),
    ]
)

sigma = agg["Sigma_Historical"].to_numpy()
sigma_p = agg["sigma_price"].to_numpy()
delta = agg["_delta_t"].to_numpy()
bs = agg["band_state_ps2"].to_numpy()
oref = agg["O_Ref"].to_numpy()
total_width = agg["total_width"].to_numpy()
c_pre = agg["C_Pre_Target_2"].to_numpy()
years_arr = np.array([s.year for s in sessions])

band_cols_list = [
    "Band_AE_Pos_Upper",
    "Band_AE_Pos_Lower",
    "Band_AE_Neg_Upper",
    "Band_AE_Neg_Lower",
    "Band_FE_Pos_Upper",
    "Band_FE_Pos_Lower",
    "Band_FE_Neg_Upper",
    "Band_FE_Neg_Lower",
]

print(f"Sessions: {n:,} ({sessions[0]} — {sessions[-1]})")


print("[1/9] Band evolution...")
years_u = sorted(set(years_arr))
yr_m_s = np.array([np.median(sigma_p[years_arr == y]) for y in years_u])
yr_m_d = np.array([np.median(delta[years_arr == y]) for y in years_u])
yr_m_w = np.array([np.median(total_width[years_arr == y]) for y in years_u])

with plt.rc_context(STY):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, data, label, color in [
        (ax1, yr_m_s, "σ·O_Ref (pts)", C["navy"]),
        (ax2, yr_m_d, "δ_t (half-width, pts)", C["coral"]),
        (ax3, yr_m_w, "AE band total width (pts)", C["teal"]),
    ]:
        ax.fill_between(years_u, data, alpha=0.12, color=color)
        ax.plot(years_u, data, color=color, linewidth=2, marker="o", markersize=3)
        ax.set_ylabel(label)
        ax.legend([f"Median {label}"], fontsize=8, loc="upper left")
        ax.set_xlim(years_u[0] - 0.5, years_u[-1] + 0.5)
        ax.grid(True, alpha=0.2)

    ax1.set_title("Yang-Zhang Volatility Scaled by Reference Price", fontweight="bold")
    ax2.set_title("Band Half-Width δ_t = k·σ·O_Ref  (k=0.1)", fontweight="bold")
    ax3.set_title(
        "Total AE Band Width (AE Pos Upper − AE Neg Lower)", fontweight="bold"
    )
    ax3.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(f"{OUT}/band_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print("[2/9] Sigma distribution...")
with plt.rc_context(STY):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(sigma * 100, bins=80, color=C["blue"], edgecolor="white", alpha=0.85)
    ax1.axvline(
        np.median(sigma) * 100,
        color=C["coral"],
        linestyle="--",
        linewidth=2,
        label=f"Median = {np.median(sigma) * 100:.2f}%",
    )
    ax1.set_xlabel("Daily Sigma (%)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Yang-Zhang Sigma Distribution", fontweight="bold")
    ax1.legend()

    dn = (delta / (sigma_p + 1e-9))[~np.isnan(delta)]
    ax2.hist(dn, bins=80, color=C["green"], edgecolor="white", alpha=0.85)
    ax2.axvline(
        np.median(dn),
        color=C["coral"],
        linestyle="--",
        linewidth=2,
        label=f"Median = {np.median(dn):.4f}",
    )
    ax2.set_xlabel("Normalized δ_t (fraction of sigma_price)")
    ax2.set_title("Normalized Band Width Distribution", fontweight="bold")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/sigma_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print("[3/9] Band level distributions...")
with plt.rc_context(STY):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    pos_bands = [
        "Band_AE_Pos_Lower",
        "Band_AE_Pos_Upper",
        "Band_FE_Pos_Lower",
        "Band_FE_Pos_Upper",
    ]
    neg_bands = [
        "Band_AE_Neg_Upper",
        "Band_AE_Neg_Lower",
        "Band_FE_Neg_Upper",
        "Band_FE_Neg_Lower",
    ]
    pos_colors = [C["lime"], C["green"], C["sky"], C["teal"]]
    neg_colors = [C["pink"], C["coral"], C["purple"], C["red"]]

    for b, clr in zip(pos_bands, pos_colors):
        rel = (agg[b].to_numpy() - oref) / (sigma_p + 1e-9)
        ax1.hist(
            rel[~np.isnan(rel)],
            bins=60,
            alpha=0.5,
            color=clr,
            label=b.replace("Band_", "").replace("_", " "),
            histtype="stepfilled",
        )
    ax1.set_xlabel("σ units from O_Ref")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Positive-Side Band Levels", fontweight="bold")
    ax1.legend(fontsize=7)
    ax1.set_xlim(-0.05, 0.35)

    for b, clr in zip(neg_bands, neg_colors):
        rel = (agg[b].to_numpy() - oref) / (sigma_p + 1e-9)
        ax2.hist(
            rel[~np.isnan(rel)],
            bins=60,
            alpha=0.5,
            color=clr,
            label=b.replace("Band_", "").replace("_", " "),
            histtype="stepfilled",
        )
    ax2.set_xlabel("σ units from O_Ref")
    ax2.set_title("Negative-Side Band Levels", fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.set_xlim(-0.35, 0.05)

    fig.tight_layout()
    fig.savefig(f"{OUT}/band_levels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print("[4/9] Zone distribution...")
zone_counts = {z: (bs == z).sum() for z in range(1, 8)}
z_labels = [
    "1\nAE Mid",
    "2\nBull",
    "3\nBear",
    "4\nFE Bull",
    "5\nFE Bear",
    "6\nExt Bull",
    "7\nExt Bear",
]

with plt.rc_context(STY):
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(
        z_labels,
        [zone_counts[z] for z in range(1, 8)],
        color=ZONE_COLORS,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, z in zip(bars, range(1, 8)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{zone_counts[z] / n * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Sessions")
    ax.set_ylim(0, max(zone_counts.values()) * 1.15)
    ax.set_title(f"Zone Distribution ({n:,} sessions)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/zone_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# Zone over time
with plt.rc_context(STY):
    fig, ax = plt.subplots(figsize=(14, 5))
    zby = np.zeros((7, len(years_u)))
    for i, y in enumerate(years_u):
        m = years_arr == y
        for z in range(1, 8):
            zby[z - 1, i] = (bs[m] == z).sum() / m.sum() * 100
    ax.stackplot(years_u, zby, labels=z_labels, colors=ZONE_COLORS, alpha=0.85)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of Sessions")
    ax.set_title("Zone Distribution Over Time", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(years_u[0], years_u[-1])
    fig.tight_layout()
    fig.savefig(f"{OUT}/zone_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print("[5/9] Band hit rates...")
hit_rates = {}
for bcol in band_cols_list:
    bv = agg[bcol].to_numpy()
    h1 = agg["H_Target_1"].to_numpy()
    l1 = agg["L_Target_1"].to_numpy()
    h2 = agg["H_Target_2"].to_numpy()
    l2 = agg["L_Target_2"].to_numpy()
    mask = ~np.isnan(bv)
    hit_rates[bcol] = {
        "up_t1": (h1[mask] >= bv[mask]).mean(),
        "down_t1": (l1[mask] <= bv[mask]).mean(),
        "up_t2": (h2[mask] >= bv[mask]).mean(),
        "down_t2": (l2[mask] <= bv[mask]).mean(),
    }

with plt.rc_context(STY):
    fig, ax = plt.subplots(figsize=(14, 5))
    names_s = [b.replace("Band_", "").replace("_", " ") for b in band_cols_list]
    x = np.arange(len(names_s))
    w = 0.2
    for i, (label, key, color) in enumerate(
        [
            ("T1 Up (High ≥ Band)", "up_t1", C["green"]),
            ("T1 Down (Low ≤ Band)", "down_t1", C["red"]),
            ("T2 Up", "up_t2", C["teal"]),
            ("T2 Down", "down_t2", C["orange"]),
        ]
    ):
        vals = [hit_rates[b][key] * 100 for b in band_cols_list]
        ax.bar(
            x + (i - 1.5) * w,
            vals,
            w,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names_s, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Band Hit Rates by Session", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/band_hit_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print("[6/9] Zone transitions...")
bs_prev = agg.with_columns(pl.col("band_state_ps2").shift(1).alias("bp"))[
    "bp"
].to_numpy()
trans = np.zeros((8, 8))
for i in range(1, n):
    if bs_prev[i] > 0 and bs[i] > 0:
        trans[int(bs_prev[i]), int(bs[i])] += 1
tp = np.zeros_like(trans)
for i in range(1, 8):
    rs = trans[i, 1:].sum()
    if rs > 0:
        tp[i, 1:] = trans[i, 1:] / rs

with plt.rc_context(STY):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(tp[1:, 1:] * 100, cmap="YlOrRd", aspect="auto", vmin=0, vmax=85)
    for i in range(7):
        for j in range(7):
            v = tp[i + 1, j + 1] * 100
            if v > 0.5:
                ax.text(
                    j,
                    i,
                    f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > 40 else "black",
                )
    ax.set_xticks(range(7))
    ax.set_xticklabels(z_labels, fontsize=8)
    ax.set_yticks(range(7))
    ax.set_yticklabels(z_labels, fontsize=8)
    ax.set_xlabel("Next Day Zone")
    ax.set_ylabel("Today's Zone")
    ax.set_title("Zone Transition Matrix (%)", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Transition Probability (%)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/zone_transitions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print("[7/9] Monte Carlo permutation...")
t1_rows = []
for sr in agg.iter_rows(named=True):
    o = sr.get("O_Target_1", 0)
    h = sr.get("H_Target_1", 0)
    l = sr.get("L_Target_1", 0)
    if not o:
        continue
    sp = (sr.get("Sigma_Historical", 0.01) or 0.01) * (sr.get("O_Ref", o) or 1)
    zp = max(0, h - o) / (sp + 1e-9)
    zn = max(0, o - l) / (sp + 1e-9)
    zd = 1 if zp > zn else 0
    t1_rows.append(zd)
z_dir_t1 = np.array(t1_rows)
true_up = z_dir_t1.mean()
n_s = len(z_dir_t1)

n_sims = 10000
shuffled = np.array([np.random.permutation(z_dir_t1).mean() for _ in range(n_sims)])

with plt.rc_context({**STY, "font.size": 11}):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(shuffled)
    x_kde = np.linspace(shuffled.min() - 0.005, shuffled.max() + 0.005, 500)
    ax.fill_between(x_kde, kde(x_kde), alpha=0.25, color=C["blue"])
    ax.plot(
        x_kde,
        kde(x_kde),
        color=C["blue"],
        linewidth=2,
        label=f"Null Distribution\n({n_sims:,} permutations)",
    )

    ax.axvline(
        true_up,
        color=C["coral"],
        linestyle="--",
        linewidth=2.5,
        label=f"Observed: {true_up * 100:.1f}%",
    )
    ax.axvline(
        0.5,
        color=C["charcoal"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.6,
        label="Null Mean (50.0%)",
    )

    nu_std = np.std(shuffled)
    cl = 0.5 - 1.96 * nu_std
    ch = 0.5 + 1.96 * nu_std
    for bnd in [cl, ch]:
        ax.axvline(bnd, color=C["gray"], linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvspan(x_kde[0], cl, alpha=0.04, color=C["coral"])
    ax.axvspan(ch, x_kde[-1], alpha=0.04, color=C["coral"], label="α=0.05 rejection")

    p_val = (np.abs(shuffled - 0.5) >= abs(true_up - 0.5)).mean()
    ax.text(
        0.98,
        0.95,
        f"n = {n_s:,} sessions\nObserved = {true_up * 100:.1f}%\n"
        f"Null σ = {nu_std * 100:.2f}%\np = {p_val:.4f}\n→ Direction is random",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=C["white"],
            alpha=0.85,
            edgecolor=C["gray"],
        ),
    )
    ax.set_xlabel("Up Rate")
    ax.set_ylabel("Density")
    ax.set_title("Monte Carlo Permutation Test — Target_1 Direction", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0.465, 0.535)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
    fig.tight_layout()
    fig.savefig(f"{OUT}/monte_carlo_direction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print("[8/9] Direction ACF...")
x = z_dir_t1 - z_dir_t1.mean()
n_lags = 20
acf_vals = np.array([np.corrcoef(x[: n_s - i], x[i:])[0, 1] for i in range(n_lags + 1)])
ci = 1.96 / np.sqrt(n_s)
lags = np.arange(1, n_lags + 1)
acf_p = acf_vals[1:]

with plt.rc_context(STY):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(lags, -ci, ci, alpha=0.12, color=C["blue"])
    ax.axhline(0, color=C["charcoal"], linewidth=0.5)
    ax.axhline(ci, color=C["gray"], linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(
        -ci,
        color=C["gray"],
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
        label=f"Bartlett 95% CI (±{ci:.3f})",
    )
    markerline, stemlines, baseline = ax.stem(
        lags, acf_p, linefmt=C["navy"], markerfmt="o", basefmt=" "
    )
    plt.setp(stemlines, linewidth=1.5)
    plt.setp(markerline, markersize=5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Direction ACF — Target_1", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-ci * 2.5, ci * 2.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/zone_duration_autocorr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

durations = []
cz = bs[0]
clen = 1
for i in range(1, n):
    if bs[i] == cz:
        clen += 1
    else:
        durations.append(clen)
        cz = bs[i]
        clen = 1
durations.append(clen)

with plt.rc_context(STY):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.hist(
        durations, bins=range(1, 25), color=C["orange"], edgecolor="white", alpha=0.85
    )
    ax.axvline(
        np.median(durations),
        color=C["navy"],
        linestyle="--",
        linewidth=2,
        label=f"Median = {np.median(durations):.0f} days",
    )
    ax.set_xlabel("Consecutive Days in Same Zone")
    ax.set_ylabel("Frequency")
    ax.set_title("Zone Duration Distribution", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/zone_duration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print("[9/9] Price reaction at bands...")
price_reactions = {}
for tlabel in ["Target_1", "Target_2"]:
    for bcol in band_cols_list:
        key = f"{tlabel}_{bcol}"
        reactions = []
        for sr in agg.iter_rows(named=True):
            bv = sr.get(bcol)
            o = sr.get(f"O_{tlabel}", 0)
            c = sr.get(f"C_{tlabel}", o) if sr.get(f"C_{tlabel}") else o
            h = sr.get(f"H_{tlabel}", o)
            l = sr.get(f"L_{tlabel}", o)
            sp = (sr.get("Sigma_Historical", 0.01) or 0.01) * (sr.get("O_Ref", o) or 1)
            if not bv or not o:
                continue
            reached_up = h >= bv
            reached_down = l <= bv
            if reached_up:
                reactions.append((c - o) / (sp + 1e-9))  # normalized final move
            if reached_down:
                reactions.append((c - o) / (sp + 1e-9))
        if reactions:
            price_reactions[key] = np.array(reactions)

with plt.rc_context(STY):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, tlabel, title in [
        (axes[0], "Target_1", "Target_1 (AM)"),
        (axes[1], "Target_2", "Target_2 (PM)"),
    ]:
        data_sets = []
        labels_list = []
        colors_list = [
            C["green"],
            C["lime"],
            C["teal"],
            C["sky"],
            C["coral"],
            C["pink"],
            C["red"],
            C["purple"],
        ]
        for bi, bcol in enumerate(band_cols_list):
            key = f"{tlabel}_{bcol}"
            if key in price_reactions and len(price_reactions[key]) > 20:
                vals = price_reactions[key]
                vals = vals[~np.isnan(vals) & (np.abs(vals) < 3)]  # clip extreme
                data_sets.append(vals)
                labels_list.append(bcol.replace("Band_", "").replace("_", " "))
        if data_sets:
            bp = ax.boxplot(
                data_sets,
                labels=labels_list,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": C["charcoal"], "linewidth": 1.5},
            )
            for patch, clr in zip(bp["boxes"], colors_list[: len(data_sets)]):
                patch.set_facecolor(clr)
                patch.set_alpha(0.6)
            ax.axhline(0, color=C["charcoal"], linewidth=0.5)
            ax.set_ylabel("(C − O) / σ_price")
            ax.set_title(title, fontweight="bold")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.suptitle(
        "Price Reaction When Session Reaches Each Band Level", fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/price_reaction_bands.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"\n{'=' * 60}")
print("ANALYSIS SUMMARY")
print(f"{'=' * 60}")
print(f"  Sessions: {n:,} | Sigma median: {np.median(sigma) * 100:.2f}% daily")
print(
    f"  Zones: {', '.join(f'{z}:{zone_counts[z] / n * 100:.1f}%' for z in range(1, 8))}"
)
print(f"  Zone duration median: {np.median(durations):.0f}d")
print(f"  T1 Up rate: {true_up * 100:.1f}% (p={p_val:.4f})")
print(f"\n  Charts: {OUT}/")
for f in sorted(Path(OUT).glob("band_*.png")):
    print(f"    {f.name}")
print("Done.")

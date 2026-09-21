import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe, works from the web app too
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- shared style config -------------------------------------------------
PRIMARY = "#4f46e5"      # indigo (matches app theme)
PRIMARY_DARK = "#3730a3"
ACCENT = "#f97316"       # warm orange accent
GRID_COLOR = "#e2e8f0"
TEXT_COLOR = "#0f172a"
MUTED_COLOR = "#64748b"
BG_COLOR = "#f8fafc"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR,
})


def compute_stats(df):
    error = df["measured_mm"] - df["actual_mm"]
    abs_error = error.abs()
    pct_error = (abs_error / df["actual_mm"]) * 100.0

    stats = {
        "n_objects": len(df),
        "mean_error_mm": error.mean(),
        "mean_abs_error_mm": abs_error.mean(),
        "std_error_mm": error.std(ddof=1),
        "rmse_mm": float(np.sqrt((error ** 2).mean())),
        "mean_pct_error": pct_error.mean(),
        "max_abs_error_mm": abs_error.max(),
        "min_abs_error_mm": abs_error.min(),
    }
    df = df.copy()
    df["error_mm"] = error
    df["abs_error_mm"] = abs_error
    df["pct_error"] = pct_error
    return df, stats


def _style_axes(ax):
    ax.set_facecolor(BG_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID_COLOR)


def make_plots(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # actual vs measured, side by side bars per object
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(df))
    width = 0.35
    _style_axes(ax)
    ax.bar(x - width / 2, df["actual_mm"], width, label="Actual (ground truth)",
           color=PRIMARY, edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + width / 2, df["measured_mm"], width, label="Measured (from image)",
           color=ACCENT, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(df["object_id"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Length (mm)", fontweight="600")
    ax.set_title("Actual vs. Measured Object Length", fontsize=14, fontweight="700",
                 color=TEXT_COLOR, pad=38)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_by_object.png"), dpi=150)
    plt.close(fig)

    # percent error per object
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    _style_axes(ax)

    mean_pct = df["pct_error"].mean()
    colors = [PRIMARY_DARK if v > mean_pct else PRIMARY for v in df["pct_error"]]
    ax.bar(x, df["pct_error"], color=colors, edgecolor="white", linewidth=0.5, zorder=3,
           width=0.6)
    ax.axhline(mean_pct, color=ACCENT, linestyle="--", linewidth=2,
               label=f"mean % error ({mean_pct:.2f}%)", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["object_id"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Percent error (%)", fontweight="600")
    ax.set_title("Percent Error per Object", fontsize=14, fontweight="700",
                 color=TEXT_COLOR, pad=38)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "percent_error.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Validate measurement accuracy against ground truth")
    parser.add_argument("--csv", required=True, help="CSV with object_id, actual_mm, measured_mm, distance_mm")
    parser.add_argument("--out", default="results", help="output folder for report + plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required_cols = {"object_id", "actual_mm", "measured_mm"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain at least these columns: {required_cols}")

    df_with_error, stats = compute_stats(df)

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "validation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Validation summary\n")
        f.write("===================\n")
        for k, v in stats.items():
            f.write(f"{k}: {v:.3f}\n" if isinstance(v, float) else f"{k}: {v}\n")
        f.write("\nPer-object detail\n")
        f.write(df_with_error.to_string(index=False))

    df_with_error.to_csv(os.path.join(args.out, "validation_detail.csv"), index=False)
    make_plots(df_with_error, args.out)

    print(f"Validated {stats['n_objects']} objects.")
    print(f"Mean absolute error: {stats['mean_abs_error_mm']:.2f} mm")
    print(f"RMSE:                {stats['rmse_mm']:.2f} mm")
    print(f"Mean percent error:  {stats['mean_pct_error']:.2f} %")
    print(f"Std dev of error:    {stats['std_error_mm']:.2f} mm")
    print(f"\nFull report written to '{args.out}/'")


if __name__ == "__main__":
    main()



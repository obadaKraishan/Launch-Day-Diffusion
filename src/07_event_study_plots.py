"""
07_event_study_plots.py
Create descriptive event-study figures from the aligned panels.

Inputs:
  - data/processed/event_windows.csv     (hn_id, owner, repo, t_day, stars_day, stars_cum_since_launch, stars_cum_abs)
  - data/processed/features_labels.csv   (labels + HN/GH features incl. is_show_hn, post_hour_utc, is_weekend_post)

Outputs:
  - outputs/figures/event_curve_overall.png
  - outputs/figures/event_curve_showhn_vs_other.png
  - outputs/figures/event_curve_weekday_vs_weekend.png
  - outputs/figures/event_curve_posthour_bins.png
  - outputs/summaries/07_event_study_plots_summary.txt

Author: <OBADA KRAISHAN>
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_FIG_DIR = "outputs/figures"
DEFAULT_SUMMARY = "outputs/summaries/07_event_study_plots_summary.txt"

def ensure_dirs(fig_dir: str, summary_txt: str):
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(summary_txt), exist_ok=True)

def coerce_bool(series):
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true","1","t","yes","y"]).astype(int)

def agg_curve(df, ycol="stars_cum_since_launch"):
    g = (df.groupby("t_day")[ycol]
           .agg(["mean","median","count"])
           .reset_index()
           .sort_values("t_day"))
    return g

def plot_curve(df_curve, title, out_png, ylab="Cumulative stars since launch"):
    plt.figure(figsize=(7.2,4.5))
    plt.plot(df_curve["t_day"], df_curve["mean"], label="Mean")
    plt.plot(df_curve["t_day"], df_curve["median"], linestyle="--", label="Median")
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlabel("Days relative to HN post (t_day)")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def bin_post_hour(h):
    try:
        h = int(h)
    except Exception:
        return "NA"
    if   0 <= h <= 5:  return "00–05"
    elif 6 <= h <= 11: return "06–11"
    elif 12 <= h <= 17:return "12–17"
    elif 18 <= h <= 23:return "18–23"
    return "NA"

def main():
    ap = argparse.ArgumentParser(description="Make event-study plots.")
    ap.add_argument("--events_csv", default="data/processed/event_windows.csv", help="Event windows CSV")
    ap.add_argument("--features_csv", default="data/processed/features_labels.csv", help="Features+labels CSV")
    ap.add_argument("--out_dir", default=DEFAULT_FIG_DIR, help="Output figures directory")
    ap.add_argument("--summary_txt", default=DEFAULT_SUMMARY, help="Where to write the textual summary")
    args = ap.parse_args()

    fig_dir = args.out_dir
    summary_txt = args.summary_txt
    ensure_dirs(fig_dir, summary_txt)

    # Load data
    if not (os.path.exists(args.events_csv) and os.path.exists(args.features_csv)):
        print("Missing input CSV(s).", file=sys.stderr)
        sys.exit(1)

    ev = pd.read_csv(args.events_csv)
    fl = pd.read_csv(args.features_csv)

    # Keys
    keycols = ["hn_id","owner","repo"]

    # Safe types
    if "t_day" in ev.columns:
        ev["t_day"] = pd.to_numeric(ev["t_day"], errors="coerce").astype("Int64")

    # Join flags from features
    join_cols = keycols + ["is_show_hn","post_hour_utc","is_weekend_post"]
    for c in join_cols:
        if c not in fl.columns:
            fl[c] = np.nan
    tmp = fl[join_cols].copy()
    tmp["is_show_hn_bin"] = coerce_bool(tmp["is_show_hn"])
    tmp["is_weekend_post_bin"] = pd.to_numeric(tmp["is_weekend_post"], errors="coerce").fillna(0).astype(int)
    tmp["post_hour_bin"] = tmp["post_hour_utc"].apply(bin_post_hour)

    panel = ev.merge(tmp[[*keycols,"is_show_hn_bin","is_weekend_post_bin","post_hour_bin"]], on=keycols, how="left")
    panel = panel.dropna(subset=["t_day","stars_cum_since_launch"])

    # ==== Overall curve ====
    overall = agg_curve(panel)
    plot_curve(overall, "Event Study: Overall", os.path.join(fig_dir, "event_curve_overall.png"))

    # ==== Show HN vs others ====
    sh = panel[panel["is_show_hn_bin"] == 1]
    non = panel[panel["is_show_hn_bin"] != 1]
    c_sh = agg_curve(sh)
    c_non = agg_curve(non)
    plt.figure(figsize=(7.2,4.5))
    plt.plot(c_sh["t_day"], c_sh["mean"], label="Show HN (mean)")
    plt.plot(c_non["t_day"], c_non["mean"], label="Other (mean)")
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlabel("Days relative to HN post (t_day)")
    plt.ylabel("Cumulative stars since launch")
    plt.title("Event Study: Show HN vs Others (mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "event_curve_showhn_vs_other.png"), dpi=150)
    plt.close()

    # ==== Weekday vs Weekend posts ====
    wk = panel[panel["is_weekend_post_bin"] == 0]
    we = panel[panel["is_weekend_post_bin"] == 1]
    c_wk = agg_curve(wk)
    c_we = agg_curve(we)
    plt.figure(figsize=(7.2,4.5))
    plt.plot(c_wk["t_day"], c_wk["mean"], label="Weekday (mean)")
    plt.plot(c_we["t_day"], c_we["mean"], label="Weekend (mean)")
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlabel("Days relative to HN post (t_day)")
    plt.ylabel("Cumulative stars since launch")
    plt.title("Event Study: Weekday vs Weekend (mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "event_curve_weekday_vs_weekend.png"), dpi=150)
    plt.close()

    # ==== Post-hour bins ====
    curves = []
    for b in ["00–05","06–11","12–17","18–23"]:
        sub = panel[panel["post_hour_bin"] == b]
        if not sub.empty:
            curves.append((b, agg_curve(sub)))

    plt.figure(figsize=(7.8,5.0))
    for label, cv in curves:
        plt.plot(cv["t_day"], cv["mean"], label=label)
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlabel("Days relative to HN post (t_day)")
    plt.ylabel("Cumulative stars since launch")
    plt.title("Event Study: Posting Hour Bins (mean)")
    plt.legend(title="UTC hour")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "event_curve_posthour_bins.png"), dpi=150)
    plt.close()

    # ==== Summary TXT ====
    def pick_at(df_curve, t):
        row = df_curve[df_curve["t_day"] == t]
        return (float(row["mean"].iloc[0]) if not row.empty else np.nan,
                float(row["median"].iloc[0]) if not row.empty else np.nan)

    m0, med0 = pick_at(overall, 0)   # 24h
    m1, med1 = pick_at(overall, 1)   # 48h
    m6, med6 = pick_at(overall, 6)   # 168h

    sh48 = c_sh[c_sh["t_day"]==1]["mean"].iloc[0] if not c_sh.empty and (c_sh["t_day"]==1).any() else np.nan
    no48 = c_non[c_non["t_day"]==1]["mean"].iloc[0] if not c_non.empty and (c_non["t_day"]==1).any() else np.nan

    lines = []
    lines.append("Event-study plots summary")
    lines.append(f"Overall Δstars (mean): 24h={m0:.1f}, 48h={m1:.1f}, 7d={m6:.1f}")
    lines.append(f"Overall Δstars (median): 24h={med0:.1f}, 48h={med1:.1f}, 7d={med6:.1f}")
    if not np.isnan(sh48) and not np.isnan(no48):
        lines.append(f"Show HN advantage at 48h (mean): {sh48:.1f} vs {no48:.1f} (diff={sh48-no48:.1f})")
    lines.append("Figures:")
    lines.append(f"- {os.path.join(fig_dir, 'event_curve_overall.png')}")
    lines.append(f"- {os.path.join(fig_dir, 'event_curve_showhn_vs_other.png')}")
    lines.append(f"- {os.path.join(fig_dir, 'event_curve_weekday_vs_weekend.png')}")
    lines.append(f"- {os.path.join(fig_dir, 'event_curve_posthour_bins.png')}")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote figures to {fig_dir} and summary to {summary_txt}")

if __name__ == "__main__":
    main()

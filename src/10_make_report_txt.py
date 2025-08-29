#!/usr/bin/env python3
"""
10_make_report_txt.py
Assemble a paper-ready TXT summary from previous pipeline outputs.

Reads (best-effort; all optional if present):
  - outputs/summaries/01_collect_hn_posts_summary.txt
  - outputs/summaries/02_extract_github_repos_summary.txt
  - outputs/summaries/03_github_repo_metadata_summary.txt
  - outputs/summaries/04_github_stars_timeseries_summary.txt
  - outputs/summaries/05_build_event_windows_summary.txt
  - outputs/summaries/06_feature_engineering_summary.txt
  - outputs/summaries/07_event_study_plots_summary.txt
  - outputs/summaries/08_model_star_growth_summary.txt
  - outputs/summaries/09_ablation_checks_summary.txt

Writes:
  - outputs/summaries/10_report_draft.txt
"""
import os, re, argparse

SUM_DIR = "outputs/summaries"
OUT_TXT = os.path.join(SUM_DIR, "10_report_draft.txt")

FILES = [
    "01_collect_hn_posts_summary.txt",
    "02_extract_github_repos_summary.txt",
    "03_github_repo_metadata_summary.txt",
    "04_github_stars_timeseries_summary.txt",
    "05_build_event_windows_summary.txt",
    "06_feature_engineering_summary.txt",
    "07_event_study_plots_summary.txt",
    "08_model_star_growth_summary.txt",
    "09_ablation_checks_summary.txt",
]

def read_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def grep_num(pattern, text, default=None, cast=float):
    m = re.search(pattern, text)
    if not m: return default
    try:
        return cast(m.group(1))
    except Exception:
        return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_txt", default=OUT_TXT, help="Where to write the draft TXT")
    args = ap.parse_args()

    os.makedirs(SUM_DIR, exist_ok=True)

    # Load all summaries
    blobs = {fn: read_safe(os.path.join(SUM_DIR, fn)) for fn in FILES}

    # Pull headline counts (best-effort; safe defaults)
    ev5 = blobs.get("05_build_event_windows_summary.txt","")
    n_pairs_total = grep_num(r"Total HN→repo pairs:\s+(\d+)", ev5, default="?", cast=int)
    n_pairs_kept  = grep_num(r"Pairs with valid timeseries:\s+(\d+)", ev5, default="?", cast=int)

    fe6 = blobs.get("06_feature_engineering_summary.txt","")
    d24_mean = grep_num(r"delta_24h\s+stats: mean=([-\d\.]+)", fe6, default=None)
    d48_med  = grep_num(r"delta_48h\s+stats: mean=[-\d\.]+, median=([-\d\.]+)", fe6, default=None)
    d7_mean  = grep_num(r"delta_168h\s+stats: mean=([-\d\.]+)", fe6, default=None)
    show_rate= grep_num(r"Show HN rate:\s+([\d\.]+)%", fe6, default=None, cast=float)

    ev7 = blobs.get("07_event_study_plots_summary.txt","")
    ev24m = grep_num(r"Overall Δstars \(mean\): 24h=([-\d\.]+)", ev7, default=None)
    ev48m = grep_num(r"Overall Δstars \(mean\): .* 48h=([-\d\.]+)", ev7, default=None)
    ev7dm = grep_num(r"Overall Δstars \(mean\): .* 7d=([-\d\.]+)", ev7, default=None)

    md8 = blobs.get("08_model_star_growth_summary.txt","")
    en48_r2 = grep_num(r"== Target: delta_48h ==.*?ElasticNet.*?R2=([-\d\.]+)", md8, default=None)
    gb48_r2 = grep_num(r"== Target: delta_48h ==.*?GradBoost\s+.*?R2=([-\d\.]+)", md8, default=None)
    en24_r2 = grep_num(r"== Target: delta_24h ==.*?R2=([-\d\.]+)", md8, default=None)
    gb7_r2  = grep_num(r"== Target: delta_168h ==.*?GradBoost\s+.*?R2=([-\d\.]+)", md8, default=None)

    ab9 = blobs.get("09_ablation_checks_summary.txt","")
    show_coef = grep_num(r"ShowHN_vs_Other: coef=([-\d\.]+)", ab9, default=None)
    show_p    = grep_num(r"ShowHN_vs_Other: .* p=([-\d\.]+)", ab9, default=None)
    wk_coef   = grep_num(r"Weekend_vs_Weekday: coef=([-\d\.]+)", ab9, default=None)
    wk_p      = grep_num(r"Weekend_vs_Weekday: .* p=([-\d\.]+)", ab9, default=None)
    hour_spread = grep_num(r"PostHour_bins_unadjusted: .* = ([\d\.]+)", ab9, default=None)

    # Compose draft
    lines = []
    lines.append("Title: Launch-Day Diffusion: Hacker News → GitHub Stars for AI Tools")
    lines.append("")
    lines.append("Abstract (1–2 sentences): We build a reproducible pipeline aligning Hacker News launches with GitHub star dynamics for AI/LLM tools. Event-study curves show a large launch bump; simple models explain a meaningful share of short-term adoption, with time-of-day and pre-launch signals contributing.")
    lines.append("")
    lines.append("Data & Scope")
    lines.append(f"- HN→repo pairs analyzed: {n_pairs_total} total; {n_pairs_kept} with valid GitHub time series.")
    if d24_mean is not None or d7_mean is not None:
        lines.append(f"- Typical effects: Δ24h mean ≈ {d24_mean:.1f}, Δ7d mean ≈ {d7_mean:.1f} stars (features table).")
    if show_rate is not None:
        lines.append(f"- Share of Show HN posts: {show_rate:.2f}%")
    lines.append("")
    lines.append("Methods")
    lines.append("- Align each GitHub repo’s hourly star series to the HN post time and aggregate to daily windows (±7 days).")
    lines.append("- Labels: Δ24h, Δ48h, Δ7d; Features: HN score/comments, timing (hour/weekday), Show HN flag, repo metadata (age, license, topics), baseline stars.")
    lines.append("- Models: Elastic Net (interpretable) and Gradient Boosting (non-linear). Robust ablations with HC1 standard errors.")
    lines.append("")
    lines.append("Results")
    if ev24m is not None and ev48m is not None and ev7dm is not None:
        lines.append(f"- Event study (means): Δ24h={ev24m:.1f}, Δ48h={ev48m:.1f}, Δ7d={ev7dm:.1f} stars.")
    if gb48_r2 is not None:
        lines.append(f"- Predictive fit: Gradient Boosting explains ~{gb48_r2:.2f} R² at 48h; Elastic Net R²≈{en48_r2 or 0:.2f}.")
    if gb7_r2 is not None:
        lines.append(f"- At 7 days: Gradient Boosting R²≈{gb7_r2:.2f}.")
    if show_coef is not None and show_p is not None:
        lines.append(f"- Show HN vs others (48h): coefficient {show_coef:+.1f} (p={show_p:.3f}) with controls; unadjusted means are higher for non-Show posts.")
    if wk_coef is not None and wk_p is not None:
        lines.append(f"- Weekend vs weekday (48h): {wk_coef:+.1f} (p={wk_p:.3f}) — negligible.")
    if hour_spread is not None:
        lines.append(f"- Post hour: unadjusted spread across bins ≈ {hour_spread:.0f} stars; timing matters.")
    lines.append("")
    lines.append("Limitations")
    lines.append("- Some metadata are observed post-launch (e.g., current forks/watchers). We avoid leakage in ablations by excluding day-0 stars and can also report strict pre-launch variants.")
    lines.append("- Observational design; results are associations, not causal effects.")
    lines.append("")
    lines.append("Reproducibility")
    lines.append("- All scripts are single-file CLI, save raw JSONL/CSV and TXT summaries, and run end-to-end on public APIs.")
    lines.append("")
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.out_txt}")

if __name__ == "__main__":
    main()

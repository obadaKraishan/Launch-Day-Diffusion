"""
09_ablation_checks.py
Paper-friendly ablations with robust SEs, avoiding label leakage by default.

Inputs:
  - data/processed/features_labels.csv

Outputs:
  - data/processed/ablation_estimates.csv
  - outputs/summaries/09_ablation_checks_summary.txt

Examples:
  # Default (target = delta_48h), no-leak features, drop launch_day_stars
  python src/09_ablation_checks.py

  # Use 7-day outcome
  python src/09_ablation_checks.py --target delta_168h

  # Allow 'momentum' feature launch_day_stars
  python src/09_ablation_checks.py --allow_launch_day

  # Be stricter: also drop post-event-ish metadata (stars_now/forks/watchers/open_issues)
  python src/09_ablation_checks.py --strict_prelaunch

  # Refit a small GB model without leaky features (reports MAE/RMSE/R2)
  python src/09_ablation_checks.py --fit_small_model

Author: <OBADA KRAISHAN>
"""
import os, sys, argparse
import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except Exception as e:
    print("statsmodels is required. Try: pip install statsmodels", file=sys.stderr)
    raise

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SUMMARY = "outputs/summaries/09_ablation_checks_summary.txt"
OUTCSV  = "data/processed/ablation_estimates.csv"

LEAK_ALWAYS = {"launch_day_stars"}  # day-0 outcome; leaks Δ24h, strongly leaks later horizons
LEAK_STRICT = {"stars_now", "forks", "watchers", "open_issues"}  # collected 'now', not at post time

def ensure_dirs():
    os.makedirs("outputs/summaries", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def hour_bin(h):
    try:
        h = int(h)
    except Exception:
        return "NA"
    if   0 <= h <= 5:  return "00–05"
    elif 6 <= h <= 11: return "06–11"
    elif 12 <= h <= 17:return "12–17"
    elif 18 <= h <= 23:return "18–23"
    return "NA"


def hour_bin_f_test(d, target, controls):
    import numpy as np
    import statsmodels.formula.api as smf
    # Build formula with controls + hour + weekday
    avail = [c for c in controls if c in d.columns]
    rhs = " + ".join(avail + ["C(post_hour_bin)", "C(post_weekday)"])
    formula = f"{target} ~ {rhs}" if rhs else f"{target} ~ C(post_hour_bin)"
    m = smf.ols(formula, data=d).fit(cov_type="HC1")
    # Joint test: all hour-bin coefficients = 0
    names = [p for p in m.params.index if p.startswith("C(post_hour_bin)[T.")]
    if names:
        R = np.zeros((len(names), len(m.params)))
        for i, nm in enumerate(names):
            R[i, list(m.params.index).index(nm)] = 1.0
        ft = m.f_test(R)
        fstat = float(np.squeeze(ft.fvalue))
        pval  = float(np.squeeze(ft.pvalue))
    else:
        fstat, pval = float("nan"), float("nan")
    g = d.groupby("post_hour_bin")[target].mean()
    return {
        "effect":"PostHour_bins_adjusted",
        "coef": float(g.max() - g.min()), "se": float("nan"),
        "pval": pval, "n": int(m.nobs),
        "mean_treat1": float(g.max()), "mean_treat0": float(g.min()),
        "formula": formula, "fstat": fstat
    }


def prep_df(df, target, allow_launch_day=False, strict_prelaunch=False):
    d = df.copy()

    # Coerce/clean
    d["is_show_hn"]      = d["is_show_hn"].astype(str).str.lower().isin(["true","1","t","yes","y"]).astype(int)
    d["is_weekend_post"] = pd.to_numeric(d["is_weekend_post"], errors="coerce").fillna(0).astype(int)
    d["post_hour_bin"]   = d["post_hour_utc"].apply(hour_bin)

    # Drop rows without target
    d = d[d[target].notna()].copy()

    # Build feature blacklist
    drop_cols = set()
    if not allow_launch_day:
        drop_cols |= LEAK_ALWAYS
    if strict_prelaunch:
        drop_cols |= LEAK_STRICT

    # Actually drop any present columns
    for c in drop_cols:
        col = c if c in d.columns else f"num__{c}"
        if col in d.columns:
            d = d.drop(columns=[col])

    return d

def ols_effect(d, target, treat_col, controls):
    """
    OLS with HC1 robust SEs:
      target ~ treat + controls + C(post_hour_bin) + C(post_weekday)
    Returns dict with coef/se/pval/n and group means.
    """
    # Make a modeling frame with controls present
    avail_controls = [c for c in controls if c in d.columns]
    formula = f"{target} ~ {treat_col}"
    if avail_controls:
        formula += " + " + " + ".join(avail_controls)
    # always include broad time-of-post controls
    if "post_hour_bin" in d.columns:
        formula += " + C(post_hour_bin)"
    if "post_weekday" in d.columns:
        formula += " + C(post_weekday)"

    model = smf.ols(formula, data=d).fit(cov_type="HC1")
    coef = model.params.get(treat_col, np.nan)
    se   = model.bse.get(treat_col, np.nan)
    pval = model.pvalues.get(treat_col, np.nan)
    n    = int(model.nobs)

    # group means (unadjusted)
    g = d.groupby(treat_col)[target].mean()
    mean0 = float(g.get(0, np.nan))
    mean1 = float(g.get(1, np.nan))
    return {
        "formula": formula,
        "coef": float(coef), "se": float(se), "pval": float(pval), "n": n,
        "mean_treat1": mean1, "mean_treat0": mean0
    }

def fit_small_model(d, target, feature_cols):
    X = d[feature_cols].copy()
    y = d[target].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = GradientBoostingRegressor(random_state=42)
    mdl.fit(X_train, y_train)
    pred = mdl.predict(X_test)
    try:
        rmse = mean_squared_error(y_test, pred, squared=False)
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_test, pred))
    }

def main():
    ap = argparse.ArgumentParser(description="Run ablation checks with robust errors and no-leak defaults.")
    ap.add_argument("--features_csv", default="data/processed/features_labels.csv")
    ap.add_argument("--target", default="delta_48h", choices=["delta_24h","delta_48h","delta_168h"])
    ap.add_argument("--allow_launch_day", action="store_true", help="Allow launch_day_stars as a feature (off by default).")
    ap.add_argument("--strict_prelaunch", action="store_true", help="Drop post-event-ish metadata (stars_now/forks/watchers/open_issues).")
    ap.add_argument("--fit_small_model", action="store_true", help="Fit a small no-leak GradBoost and report metrics.")
    args = ap.parse_args()

    ensure_dirs()
    if not os.path.exists(args.features_csv):
        print(f"Missing: {args.features_csv}", file=sys.stderr); sys.exit(1)

    df = pd.read_csv(args.features_csv)
    d  = prep_df(df, args.target, allow_launch_day=args.allow_launch_day, strict_prelaunch=args.strict_prelaunch)

    # Controls that are safe and interpretable
    controls = ["baseline_stars","hn_score","repo_age_days","title_len","is_weekend_post"]

    rows = []

    # 1) Show HN effect
    if "is_show_hn" in d.columns:
        r = ols_effect(d, args.target, "is_show_hn", controls)
        r.update({"effect":"ShowHN_vs_Other"})
        rows.append(r)

    # 2) Weekend vs Weekday
    if "is_weekend_post" in d.columns:
        r = ols_effect(d, args.target, "is_weekend_post", [c for c in controls if c != "is_weekend_post"])
        r.update({"effect":"Weekend_vs_Weekday"})
        rows.append(r)

    # 3) Posting hour (bins) — report range of bin effects vs a reference
    if "post_hour_bin" in d.columns:
        # Reference bin is whatever pandas picks first after dummying
        # We report min/max predicted difference across bins (unadjusted)
        g = d.groupby("post_hour_bin")[args.target].mean().sort_index()
        rows.append({
            "effect":"PostHour_bins_unadjusted",
            "coef": float(g.max() - g.min()),
            "se": np.nan, "pval": np.nan, "n": int(len(d)),
            "mean_treat1": float(g.max()), "mean_treat0": float(g.min()),
            "formula": "group means by post_hour_bin"
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTCSV, index=False)

    # Optional: small no-leak model
    small_model_line = ""
    if args.fit_small_model:
        feature_cols = [c for c in ["hn_score","baseline_stars","repo_age_days","title_len","is_show_hn","is_weekend_post","post_weekday","post_hour_utc"]
                        if c in d.columns]
        if feature_cols:
            m = fit_small_model(d, args.target, feature_cols)
            small_model_line = f"No-leak small GradBoost on {args.target}: MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}"

    # Summary TXT
    lines = [f"Ablations on target={args.target}  (allow_launch_day={args.allow_launch_day}, strict_prelaunch={args.strict_prelaunch})",
             ""]
    for r in rows:
        name = r["effect"]
        coef, se, pval, n = r["coef"], r["se"], r["pval"], r["n"]
        if np.isnan(se):
            lines.append(f"{name}: unadjusted max-min across bins = {coef:.2f} (n={n})")
        else:
            lines.append(f"{name}: coef={coef:.2f} (SE={se:.2f}, p={pval:.4f}, n={n})")
            lines.append(f"    Group means (unadjusted): treat= {r['mean_treat1']:.2f} vs control= {r['mean_treat0']:.2f}")
            lines.append(f"    Formula: {r['formula']}")
        lines.append("")
    if small_model_line:
        lines.append(small_model_line)

    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {OUTCSV} and {SUMMARY}")

if __name__ == "__main__":
    main()

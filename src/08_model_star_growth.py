#!/usr/bin/env python3
"""
08_model_star_growth.py
Model short-term GitHub star growth after HN launch.

Inputs:
  - data/processed/features_labels.csv   (from 06)
Outputs:
  - data/processed/model_predictions.csv
  - outputs/summaries/08_model_star_growth_summary.txt
  - outputs/figures/08_enet_coefficients_*.png
  - outputs/figures/08_gb_permutation_importance_*.png
"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

SUMMARY_TXT = "outputs/summaries/08_model_star_growth_summary.txt"
PRED_CSV    = "data/processed/model_predictions.csv"
FIG_COEF    = "outputs/figures/08_enet_coefficients.png"
FIG_IMP     = "outputs/figures/08_gb_permutation_importance.png"

def ensure_dirs():
    os.makedirs("outputs/summaries", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def make_ohe():
    """Return an OneHotEncoder that works across sklearn versions."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def pick_columns(df):
    cat_cols = [c for c in ["owner_type","license_spdx"] if c in df.columns]
    bool_cols = [c for c in [
        "is_show_hn","is_weekend_post",
        "has_numbers","has_exclamation","title_has_release","title_has_paper",
        "has_releases",
        "topics_has_llm","topics_has_rag","topics_has_transformers","topics_has_llama",
        "topics_has_agent","topics_has_vector","topics_has_diffusion","ai_topic_any"
    ] if c in df.columns]
    num_cols = [c for c in [
        "hn_score","hn_comments","post_hour_utc","post_weekday","title_len",
        "repo_age_days","readme_len","stars_now","forks","open_issues","watchers",
        "baseline_stars","launch_day_stars"
    ] if c in df.columns]
    X_cols = cat_cols + bool_cols + num_cols
    return cat_cols, bool_cols, num_cols, X_cols

def build_preprocessor(cat_cols, bool_cols, num_cols):
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  make_ohe())
    ])
    booleans = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent"))
    ])
    nums = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler())
    ])
    pre = ColumnTransformer([
        ("cat",  cat,      cat_cols),
        ("bool", booleans, bool_cols),
        ("num",  nums,     num_cols)
    ], remainder="drop")
    return pre

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def plot_top(series, title, out_png, top_k=20):
    s = series.dropna().sort_values(ascending=False).head(top_k)[::-1]
    plt.figure(figsize=(7.5, 6))
    plt.barh(s.index, s.values)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def model_one_target(df, target_col, random_state=42):
    d = df[df[target_col].notna()].copy()
    if d.empty:
        return None

    cat_cols, bool_cols, num_cols, X_cols = pick_columns(d)
    if not X_cols:
        return None

    y = d[target_col].astype(float)
    X = d[X_cols].copy()

    key_cols = [k for k in ["hn_id","owner","repo"] if k in d.columns]
    ids = d[key_cols].copy()

    pre = build_preprocessor(cat_cols, bool_cols, num_cols)

    enet = Pipeline([
        ("pre", pre),
        ("mdl", ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9],
                             alphas=np.logspace(-3,2,20),
                             cv=5, max_iter=5000, random_state=random_state))
    ])

    gbr = Pipeline([
        ("pre", pre),
        ("mdl", GradientBoostingRegressor(random_state=random_state))
    ])

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=random_state
    )

    enet.fit(X_train, y_train)
    gbr.fit(X_train, y_train)

    pred_enet = enet.predict(X_test)
    pred_gbr  = gbr.predict(X_test)

    m_enet = metrics(y_test, pred_enet)
    m_gbr  = metrics(y_test, pred_gbr)

    feat_names = enet.named_steps["pre"].get_feature_names_out()

    coef = pd.Series(enet.named_steps["mdl"].coef_, index=feat_names)
    coef_abs = coef.abs().sort_values(ascending=False)

    pi = permutation_importance(
        gbr, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
    )
    imp = pd.Series(pi.importances_mean, index=feat_names).sort_values(ascending=False)

    plot_top(coef_abs, f"Elastic Net | {target_col} | |coef| (top 20)",
             FIG_COEF.replace(".png", f"_{target_col}.png"))
    plot_top(imp, f"GradBoost | {target_col} | Permutation importance (top 20)",
             FIG_IMP.replace(".png", f"_{target_col}.png"))

    preds = ids_test.copy()
    preds[f"{target_col}_true"] = y_test.values
    preds[f"{target_col}_enet"] = pred_enet
    preds[f"{target_col}_gbr"]  = pred_gbr

    summary = {
        "target": target_col,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "enet": {"MAE": m_enet[0], "RMSE": m_enet[1], "R2": m_enet[2],
                 "best_alpha": float(enet.named_steps["mdl"].alpha_),
                 "best_l1_ratio": float(enet.named_steps["mdl"].l1_ratio_)},
        "gbr":  {"MAE": m_gbr[0],  "RMSE": m_gbr[1],  "R2": m_gbr[2]},
        "enet_top10": coef_abs.head(10).to_dict(),
        "gbr_top10":  imp.head(10).to_dict()
    }
    return preds, summary

def main():
    parser = argparse.ArgumentParser(description="Model star growth after HN launch.")
    parser.add_argument("--features_csv", default="data/processed/features_labels.csv")
    parser.add_argument("--out_pred_csv", default=PRED_CSV)
    parser.add_argument("--summary_txt",  default=SUMMARY_TXT)
    args = parser.parse_args()

    ensure_dirs()

    if not os.path.exists(args.features_csv):
        print(f"Missing input: {args.features_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.features_csv)

    targets = ["delta_24h","delta_48h","delta_168h"]
    all_preds, summaries = [], []

    for t in targets:
        res = model_one_target(df, t, random_state=42)
        if res is None:
            continue
        preds, summ = res
        all_preds.append(preds)
        summaries.append(summ)

    if all_preds:
        out = all_preds[0]
        for p in all_preds[1:]:
            out = out.merge(p, on=[c for c in ["hn_id","owner","repo"] if c in out.columns], how="outer")
        out.to_csv(args.out_pred_csv, index=False)

    lines = []
    for s in summaries:
        lines.append(f"== Target: {s['target']} ==")
        lines.append(f"Train={s['n_train']}  Test={s['n_test']}")
        lines.append(f"ElasticNet  MAE={s['enet']['MAE']:.2f}  RMSE={s['enet']['RMSE']:.2f}  R2={s['enet']['R2']:.3f}  (alpha={s['enet']['best_alpha']:.4f}, l1_ratio={s['enet']['best_l1_ratio']:.2f})")
        lines.append(f"GradBoost   MAE={s['gbr']['MAE']:.2f}   RMSE={s['gbr']['RMSE']:.2f}   R2={s['gbr']['R2']:.3f}")
        lines.append("ElasticNet top-10 |coef|:")
        for k,v in list(s["enet_top10"].items()):
            lines.append(f"  - {k}: {v:.4f}")
        lines.append("GradBoost top-10 permutation importance:")
        for k,v in list(s["gbr_top10"].items()):
            lines.append(f"  - {k}: {v:.4f}")
        lines.append("")
    with open(args.summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote predictions to {args.out_pred_csv}")
    print(f"Wrote summary to {args.summary_txt}")
    print("Saved coefficient and importance plots to outputs/figures/")

if __name__ == "__main__":
    main()

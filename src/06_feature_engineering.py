#!/usr/bin/env python3
"""
06_feature_engineering.py
Build modeling table by joining HN posts, GitHub metadata, and event windows.

Inputs:
  - data/processed/hn_posts.csv
  - data/processed/github_repos_from_hn.csv
  - data/processed/github_repos_metadata.csv
  - data/processed/event_windows.csv

Outputs:
  - data/processed/features_labels.csv
  - outputs/summaries/06_feature_engineering_summary.txt

Labels:
  - delta_24h   := stars gained within 24h after HN post  (event_windows.t_day == 0, stars_cum_since_launch)
  - delta_48h   := stars gained within 48h after HN post  (t_day == 1)
  - delta_168h  := stars gained within 168h (7 days)      (t_day == 6)
Also provides:
  - baseline_stars := absolute cumulative stars at end of day -1 (pre-launch), from event_windows.stars_cum_abs (t_day == -1)
  - launch_day_stars := stars_day at t_day == 0

Features include:
  - HN: score, comments, is_show_hn, post_hour_utc, post_weekday, is_weekend_post,
        title_len, has_numbers, has_exclamation, title_has_release, title_has_paper
  - GH: owner_type, repo_age_days, readme_len, license_spdx, has_releases,
        stars_now, forks, open_issues, watchers,
        topic flags from topics_json (llm, rag, transformers, llama, agent, vector, diffusion),
        name/topic/title composite flag ai_topic_any

Usage:
  python src/06_feature_engineering.py \
    --hn_csv data/processed/hn_posts.csv \
    --map_csv data/processed/github_repos_from_hn.csv \
    --meta_csv data/processed/github_repos_metadata.csv \
    --events_csv data/processed/event_windows.csv \
    --out_csv data/processed/features_labels.csv
"""
import os, sys, re, json, argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd

SUMMARY = "outputs/summaries/06_feature_engineering_summary.txt"

AI_KEYS = {"llm","gpt","rag","transformers","langchain","llama","mistral","agent","agents","vector","diffusion"}

def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/summaries", exist_ok=True)

def to_utc_ts(s):
    if pd.isna(s):
        return None
    try:
        return pd.to_datetime(s, utc=True).to_pydatetime()
    except Exception:
        return None

def text_has_numbers(s):
    return bool(re.search(r"\d", s or ""))

def text_has_excl(s):
    return "!" in (s or "")

def text_has_release(s):
    s = (s or "").lower()
    return any(w in s for w in ["release", "v", "v.", "version "])

def text_has_paper(s):
    s = (s or "").lower()
    return "paper" in s

def parse_topics(topics_json):
    try:
        arr = json.loads(topics_json) if isinstance(topics_json, str) else (topics_json or [])
        if not isinstance(arr, list):
            return []
        return [str(t).lower().strip() for t in arr if isinstance(t, (str,int))]
    except Exception:
        return []

def any_key_in(container, keys):
    hay = set([str(x).lower() for x in container])
    return any(k in hay for k in keys)

def build_labels(event_windows):
    # pivot-like extraction for required t_day values per (hn_id, owner, repo)
    sub = event_windows[["hn_id","owner","repo","t_day","stars_cum_since_launch","stars_cum_abs","stars_day"]].copy()
    keys = ["hn_id","owner","repo"]

    # helper to pull value at a given t_day
    def pick_at(df, day, colname, newname):
        m = df[df["t_day"] == day][keys + [colname]].rename(columns={colname: newname})
        return m

    l24 = pick_at(sub, 0,  "stars_cum_since_launch", "delta_24h")
    l48 = pick_at(sub, 1,  "stars_cum_since_launch", "delta_48h")
    l168= pick_at(sub, 6,  "stars_cum_since_launch", "delta_168h")
    base= pick_at(sub, -1, "stars_cum_abs",           "baseline_stars")
    lday= pick_at(sub, 0,  "stars_day",               "launch_day_stars")

    out = l24.merge(l48, on=keys, how="outer") \
             .merge(l168, on=keys, how="outer") \
             .merge(base, on=keys, how="left") \
             .merge(lday, on=keys, how="left")
    return out

def main():
    ap = argparse.ArgumentParser(description="Create feature & label table for modeling.")
    ap.add_argument("--hn_csv", default="data/processed/hn_posts.csv", help="HN posts CSV")
    ap.add_argument("--map_csv", default="data/processed/github_repos_from_hn.csv", help="HN→GitHub mapping CSV")
    ap.add_argument("--meta_csv", default="data/processed/github_repos_metadata.csv", help="GitHub repo metadata CSV")
    ap.add_argument("--events_csv", default="data/processed/event_windows.csv", help="Event windows CSV")
    ap.add_argument("--out_csv", default="data/processed/features_labels.csv", help="Output features+labels CSV")
    args = ap.parse_args()

    ensure_dirs()

    # Load inputs
    for p in [args.hn_csv, args.map_csv, args.meta_csv, args.events_csv]:
        if not os.path.exists(p):
            print(f"Missing input: {p}", file=sys.stderr)
            sys.exit(1)

    hn = pd.read_csv(args.hn_csv)
    mp = pd.read_csv(args.map_csv)
    meta = pd.read_csv(args.meta_csv)
    ev = pd.read_csv(args.events_csv)

    # Normalize keys
    for df in (mp, meta, ev):
        if "owner" in df.columns:
            df["owner"] = df["owner"].astype(str).str.lower().str.strip()
        if "repo" in df.columns:
            df["repo"] = df["repo"].astype(str).str.lower().str.strip()

    # HN features
    hn["post_time_utc"] = hn["time_utc"].apply(to_utc_ts)
    hn["post_hour_utc"] = pd.to_datetime(hn["time_utc"], utc=True, errors="coerce").dt.hour
    hn["post_weekday"]  = pd.to_datetime(hn["time_utc"], utc=True, errors="coerce").dt.weekday  # 0=Mon
    hn["is_weekend_post"] = hn["post_weekday"].isin([5,6]).astype(int)
    hn["title_len"] = hn["title"].astype(str).str.len()
    hn["has_numbers"] = hn["title"].apply(text_has_numbers).astype(int)
    hn["has_exclamation"] = hn["title"].apply(text_has_excl).astype(int)
    hn["title_has_release"] = hn["title"].apply(text_has_release).astype(int)
    hn["title_has_paper"]   = hn["title"].apply(text_has_paper).astype(int)

    hn_feats = hn[["hn_id","time_utc","post_time_utc","post_hour_utc","post_weekday","is_weekend_post",
                   "title","title_len","has_numbers","has_exclamation","title_has_release","title_has_paper",
                   "score","descendants","is_show_hn","author"]].copy()
    hn_feats = hn_feats.rename(columns={
        "score":"hn_score",
        "descendants":"hn_comments"
    })

    # Map (HN→repo) — keep each unique pair
    mp2 = mp[["hn_id","owner","repo"]].dropna().drop_duplicates()

    # GitHub metadata features
    # Parse repo created_at, compute age at post time later (needs post_time)
    meta_feats = meta.copy()
    for col in ["created_at","pushed_at"]:
        if col in meta_feats.columns:
            meta_feats[col] = pd.to_datetime(meta_feats[col], utc=True, errors="coerce")

    # Topic flags
    topic_flags = []
    for idx, row in meta_feats.iterrows():
        topics = parse_topics(row.get("topics_json", "[]"))
        flags = {
            "topics_has_llm":        int(any(k in topics for k in ["llm","llms"])),
            "topics_has_rag":        int("rag" in topics),
            "topics_has_transformers": int("transformers" in topics),
            "topics_has_llama":      int("llama" in topics or "llama2" in topics or "llama3" in topics),
            "topics_has_agent":      int(any(k in topics for k in ["agent","agents"])),
            "topics_has_vector":     int("vector" in topics or "vector-store" in topics),
            "topics_has_diffusion":  int("diffusion" in topics),
        }
        topic_flags.append(flags)
    topic_df = pd.DataFrame(topic_flags)
    meta_feats = pd.concat([meta_feats.reset_index(drop=True), topic_df], axis=1)

    # Composite AI flag using topics + repo name + HN title
    meta_feats["repo_name_str"] = (meta_feats["owner"].astype(str) + "/" + meta_feats["repo"].astype(str)).str.lower()

    # Merge mapping → HN (we need post_time for age)
    base = mp2.merge(hn_feats, on="hn_id", how="left")

    # Now merge GH meta
    use_cols = ["owner","repo","owner_type","created_at","pushed_at","license_spdx",
                "readme_len","has_releases","stars_now","forks","open_issues","watchers",
                "topics_has_llm","topics_has_rag","topics_has_transformers","topics_has_llama",
                "topics_has_agent","topics_has_vector","topics_has_diffusion","repo_name_str"]
    meta_small = meta_feats[use_cols].copy()
    base = base.merge(meta_small, on=["owner","repo"], how="left")

    # repo_age_days at post time
    base["repo_age_days"] = np.where(
        base["created_at"].notna() & base["post_time_utc"].notna(),
        (pd.to_datetime(base["post_time_utc"], utc=True) - base["created_at"]).dt.days,
        np.nan
    )

    # any AI keyword match across topics OR repo name OR HN title
    def make_ai_any(row):
        title = str(row.get("title","")).lower()
        name = str(row.get("repo_name_str","")).lower()
        topic_hit = any([
            row.get("topics_has_llm",0), row.get("topics_has_rag",0), row.get("topics_has_transformers",0),
            row.get("topics_has_llama",0), row.get("topics_has_agent",0), row.get("topics_has_vector",0),
            row.get("topics_has_diffusion",0)
        ])
        name_hit = any(k in name for k in AI_KEYS)
        title_hit = any(k in title for k in AI_KEYS)
        return int(topic_hit or name_hit or title_hit)

    base["ai_topic_any"] = base.apply(make_ai_any, axis=1)

    # Labels from event windows
    ev["owner"] = ev["owner"].astype(str).str.lower().str.strip()
    ev["repo"]  = ev["repo"].astype(str).str.lower().str.strip()
    labels = build_labels(ev)

    out = base.merge(labels, on=["hn_id","owner","repo"], how="left")

    # Final selection & ordering
    cols = [
        # keys
        "hn_id","owner","repo","time_utc","post_time_utc",
        # labels
        "delta_24h","delta_48h","delta_168h","baseline_stars","launch_day_stars",
        # HN features
        "hn_score","hn_comments","is_show_hn","post_hour_utc","post_weekday","is_weekend_post",
        "title_len","has_numbers","has_exclamation","title_has_release","title_has_paper",
        # GH features
        "owner_type","repo_age_days","readme_len","license_spdx","has_releases",
        "stars_now","forks","open_issues","watchers",
        "topics_has_llm","topics_has_rag","topics_has_transformers","topics_has_llama",
        "topics_has_agent","topics_has_vector","topics_has_diffusion","ai_topic_any",
    ]
    # ensure columns exist
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]

    # Save
    out.to_csv(args.out_csv, index=False)

    # Summary TXT
    lines = []
    n_rows = len(out)
    cov24 = out["delta_24h"].notna().mean() if n_rows else 0
    cov48 = out["delta_48h"].notna().mean() if n_rows else 0
    cov168= out["delta_168h"].notna().mean() if n_rows else 0

    def safe_stats(s):
        s = pd.to_numeric(s, errors="coerce")
        return pd.Series({
            "mean": float(s.mean(skipna=True)) if len(s) else np.nan,
            "median": float(s.median(skipna=True)) if len(s) else np.nan,
            "p90": float(s.quantile(0.90)) if len(s) else np.nan
        })

    stats24 = safe_stats(out["delta_24h"])
    stats48 = safe_stats(out["delta_48h"])
    stats168= safe_stats(out["delta_168h"])

    lines.append(f"Rows (HN→repo pairs): {n_rows}")
    lines.append(f"Coverage delta_24h:  {cov24:.2%}")
    lines.append(f"Coverage delta_48h:  {cov48:.2%}")
    lines.append(f"Coverage delta_168h: {cov168:.2%}")
    lines.append(f"delta_24h  stats: mean={stats24['mean']:.2f}, median={stats24['median']:.2f}, p90={stats24['p90']:.2f}")
    lines.append(f"delta_48h  stats: mean={stats48['mean']:.2f}, median={stats48['median']:.2f}, p90={stats48['p90']:.2f}")
    lines.append(f"delta_168h stats: mean={stats168['mean']:.2f}, median={stats168['median']:.2f}, p90={stats168['p90']:.2f}")
    lines.append(f"Show HN rate: {(out['is_show_hn']==True).mean():.2%}")
    lines.append(f"Weekend post rate: {(out['is_weekend_post']==1).mean():.2%}")
    lines.append(f"AI topic (any) rate: {(out['ai_topic_any']==1).mean():.2%}")

    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.out_csv} and {SUMMARY}")

if __name__ == "__main__":
    main()

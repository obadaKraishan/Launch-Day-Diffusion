#!/usr/bin/env python3
"""
05_build_event_windows.py
Align GitHub star time series to Hacker News post time and build +/- windowed daily event panels.

Author: <OBADA KRAISHAN>
"""
import os, sys, argparse
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

SUMMARY = "outputs/summaries/05_build_event_windows_summary.txt"

def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/summaries", exist_ok=True)

def to_utc_timestamp(s):
    """Parse ISO (with or without TZ) to aware UTC datetime."""
    if pd.isna(s):
        return None
    try:
        dt = pd.to_datetime(s, utc=True)
        return dt.to_pydatetime()
    except Exception:
        return None

def merge_asof_last(df_ts, when_dt):
    """
    Given a repo timeseries df (columns: ts_hour [datetime64[ns, UTC]], stars_cum),
    return the last known 'stars_cum' at or before 'when_dt'. If none, return 0.
    """
    if df_ts.empty or when_dt is None:
        return 0.0
    # when_dt is already timezone-aware → do NOT pass tz= again
    tmp = pd.DataFrame({"when": [pd.Timestamp(when_dt)]})
    joined = pd.merge_asof(
        tmp.sort_values("when"),
        df_ts.sort_values("ts_hour")[["ts_hour", "stars_cum"]],
        left_on="when", right_on="ts_hour", direction="backward"
    )
    val = joined["stars_cum"].iloc[0]
    return float(val) if pd.notnull(val) else 0.0

def build_event_panel_for_pair(repo_ts, post_dt_utc, window_days):
    """
    repo_ts: DataFrame for a single (owner,repo), columns: ts_hour (UTC), stars_hourly, stars_cum
    post_dt_utc: timezone-aware (UTC) datetime of the HN post
    Returns DataFrame with t_day in [-window_days .. +window_days] and:
      - stars_day: sum of stars_hourly within that relative day bucket
      - stars_cum_since_launch: cumulative stars since just before post time
      - stars_cum_abs: absolute cumulative stars at day end
    """
    if repo_ts.empty or post_dt_utc is None:
        return pd.DataFrame(columns=["t_day","stars_day","stars_cum_since_launch","stars_cum_abs"])

    # Ensure timezone-aware ts_hour
    if repo_ts["ts_hour"].dt.tz is None:
        repo_ts["ts_hour"] = repo_ts["ts_hour"].dt.tz_localize("UTC")

    # t_day = floor((ts_hour - post_time) / 24h)
    rel_hours = (repo_ts["ts_hour"] - pd.Timestamp(post_dt_utc)) / pd.Timedelta(hours=1)
    repo_ts = repo_ts.assign(t_day=np.floor(rel_hours / 24.0).astype(int))

    # keep only rows inside the window +/- window_days
    mask = (repo_ts["t_day"] >= -window_days) & (repo_ts["t_day"] <= window_days)
    ts_win = repo_ts.loc[mask].copy()

    # aggregate to daily stars
    stars_day = (
        ts_win.groupby("t_day")["stars_hourly"]
        .sum()
        .reindex(range(-window_days, window_days + 1), fill_value=0)
        .rename("stars_day")
        .reset_index()
    )

    # baseline: cumulative just BEFORE t=0 (i.e., last ts_hour < post_time)
    baseline_cum = merge_asof_last(repo_ts[["ts_hour","stars_cum"]], post_dt_utc - timedelta(seconds=1))

    # cumulative ABS at end of each relative day:
    day_ends = [post_dt_utc + timedelta(days=d + 1) for d in range(-window_days, window_days + 1)]
    end_vals = []
    for de in day_ends:
        val = merge_asof_last(repo_ts[["ts_hour","stars_cum"]], de - timedelta(seconds=1))
        end_vals.append(val)
    stars_cum_abs = pd.Series(end_vals, name="stars_cum_abs")

    # since launch (relative to baseline)
    stars_cum_since = stars_cum_abs - baseline_cum

    out = pd.DataFrame({
        "t_day": range(-window_days, window_days + 1),
        "stars_day": stars_day["stars_day"].values,
        "stars_cum_since_launch": stars_cum_since.values,
        "stars_cum_abs": stars_cum_abs.values
    })
    return out

def main():
    ap = argparse.ArgumentParser(description="Build +/- windowed daily event panels around HN post time.")
    ap.add_argument("--hn_csv", default="data/processed/hn_posts.csv", help="HN posts CSV")
    ap.add_argument("--map_csv", default="data/processed/github_repos_from_hn.csv", help="HN→GitHub mapping CSV")
    ap.add_argument("--stars_csv", default="data/processed/stars_timeseries.csv", help="Stars hourly timeseries CSV")
    ap.add_argument("--out_csv", default="data/processed/event_windows.csv", help="Output CSV for event windows")
    ap.add_argument("--window_days", type=int, default=7, help="Half-window in days (produces [-W .. +W])")
    args = ap.parse_args()

    ensure_dirs()

    # Load inputs
    if not (os.path.exists(args.hn_csv) and os.path.exists(args.map_csv) and os.path.exists(args.stars_csv)):
        print("Missing one or more inputs. Check --hn_csv, --map_csv, --stars_csv.", file=sys.stderr)
        sys.exit(1)

    hn = pd.read_csv(args.hn_csv)
    mp = pd.read_csv(args.map_csv)
    ts = pd.read_csv(args.stars_csv)

    # Parse/normalize times
    hn["post_time_utc"] = hn["time_utc"].apply(to_utc_timestamp)
    ts["ts_hour"] = pd.to_datetime(ts["ts_hour"], utc=True)

    # normalize owner/repo to lowercase
    if "owner" in mp.columns: mp["owner"] = mp["owner"].astype(str).str.lower().str.strip()
    if "repo"  in mp.columns: mp["repo"]  = mp["repo"].astype(str).str.lower().str.strip()
    if "owner" in ts.columns: ts["owner"] = ts["owner"].astype(str).str.lower().str.strip()
    if "repo"  in ts.columns: ts["repo"]  = ts["repo"].astype(str).str.lower().str.strip()

    # Join mapping → HN times (keep each (hn_id, owner, repo) pair)
    mp2 = mp[["hn_id","owner","repo"]].dropna().drop_duplicates()
    merged = mp2.merge(hn[["hn_id","post_time_utc"]], on="hn_id", how="left")

    out_rows = []
    have_ts_pairs = set((ts["owner"] + "/" + ts["repo"]).unique())

    for hn_id, owner, repo, post_time in merged[["hn_id","owner","repo","post_time_utc"]].itertuples(index=False):
        if pd.isna(post_time):
            continue
        key = f"{owner}/{repo}"
        if key not in have_ts_pairs:
            continue

        sub = ts[(ts["owner"] == owner) & (ts["repo"] == repo)][["ts_hour","stars_hourly","stars_cum"]].sort_values("ts_hour")
        panel = build_event_panel_for_pair(sub, post_time, args.window_days)
        if panel.empty:
            continue

        panel.insert(0, "repo", repo)
        panel.insert(0, "owner", owner)
        panel.insert(0, "post_time_utc", pd.Timestamp(post_time).isoformat())
        panel.insert(0, "hn_id", hn_id)
        out_rows.append(panel)

    if out_rows:
        out_df = pd.concat(out_rows, ignore_index=True)
    else:
        out_df = pd.DataFrame(columns=["hn_id","post_time_utc","owner","repo","t_day","stars_day","stars_cum_since_launch","stars_cum_abs"])

    out_df.to_csv(args.out_csv, index=False)

    # Summary
    n_pairs_total = len(merged)
    n_pairs_kept = out_df[["hn_id","owner","repo"]].drop_duplicates().shape[0] if not out_df.empty else 0
    post_mask = (out_df["t_day"] >= 0) if not out_df.empty else pd.Series([], dtype=bool)
    n_pairs_with_poststars = out_df.loc[post_mask].groupby(["hn_id","owner","repo"])["stars_day"].sum()
    n_pairs_with_poststars = (n_pairs_with_poststars > 0).sum() if not out_df.empty else 0

    lines = [
        f"Total HN→repo pairs: {n_pairs_total}",
        f"Pairs with valid timeseries: {n_pairs_kept}",
        f"Pairs with any post-event stars (t_day>=0): {n_pairs_with_poststars}",
        f"Window: [-{args.window_days} .. +{args.window_days}] days",
        f"Output CSV: {args.out_csv}",
    ]
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.out_csv} and {SUMMARY}")

if __name__ == "__main__":
    main()

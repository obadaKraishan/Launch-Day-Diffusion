"""
04_github_stars_timeseries.py
Fetch per-user star events with timestamps and build an HOURLY cumulative time series.

Saves:
  - data/raw/stargazers_<owner>_<repo>.jsonl    (one line per star event with 'starred_at')
  - data/processed/stars_timeseries.csv         (owner,repo,ts_hour,stars_hourly,stars_cum)
  - outputs/summaries/04_github_stars_timeseries_summary.txt

Usage:
  python src/04_github_stars_timeseries.py --input_csv data/processed/github_repos_from_hn.csv

Author: <OBADA KRAISHAN>
"""
import os, sys, json, time, argparse, math
from datetime import datetime, timezone
from typing import Dict, Any, List
import requests
import pandas as pd
from dotenv import load_dotenv

OUT_TS_CSV = "data/processed/stars_timeseries.csv"
SUMMARY    = "outputs/summaries/04_github_stars_timeseries_summary.txt"

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/summaries", exist_ok=True)

def gh_headers(token: str = "", star_mode=False) -> Dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "icwsm-hn-github-stars"
    }
    if star_mode:
        # needed to get 'starred_at' timestamps
        h["Accept"] = "application/vnd.github.star+json"
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def backoff_sleep(attempt: int):
    time.sleep(min(60, 1.5 ** attempt))

def fetch_stargazers(owner: str, repo: str, token: str, out_jsonl: str):
    """Fetch all stargazers w/ timestamps. Writes to JSONL incrementally (resume-safe)."""
    headers = gh_headers(token, star_mode=True)
    per_page = 100
    page = 1
    total = 0

    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/stargazers"
        params = {"per_page": per_page, "page": page}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 404:
            # repo might be gone/private
            break
        if r.status_code == 200:
            items = r.json()
            if not items:
                break
            with open(out_jsonl, "a", encoding="utf-8") as fw:
                for it in items:
                    # structure varies: expect 'starred_at' at top-level
                    ev = {
                        "owner": owner,
                        "repo": repo,
                        "starred_at": it.get("starred_at"),
                        "user_login": (it.get("user") or {}).get("login") if isinstance(it.get("user"), dict) else None
                    }
                    fw.write(json.dumps(ev) + "\n")
                    total += 1
            page += 1
            time.sleep(0.2)
            continue
        if r.status_code in (403, 429, 502, 503):
            backoff_sleep(page)  # mild backoff
            continue
        # other errors
        break
    return total

def build_hourly_timeseries(owner: str, repo: str, jsonl_path: str) -> pd.DataFrame:
    if not os.path.exists(jsonl_path):
        return pd.DataFrame(columns=["owner","repo","ts_hour","stars_hourly","stars_cum"])
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
                ts = ev.get("starred_at")
                if not ts:
                    continue
                # normalize to hour
                dt = datetime.fromisoformat(ts.replace("Z","+00:00")).astimezone(timezone.utc)
                hour = dt.replace(minute=0, second=0, microsecond=0)
                rows.append({"owner": owner, "repo": repo, "ts_hour": hour})
            except Exception:
                continue
    if not rows:
        return pd.DataFrame(columns=["owner","repo","ts_hour","stars_hourly","stars_cum"])

    df = pd.DataFrame(rows)
    g = df.groupby(["owner","repo","ts_hour"]).size().reset_index(name="stars_hourly")
    g = g.sort_values(["owner","repo","ts_hour"])
    # cumulative
    g["stars_cum"] = g.groupby(["owner","repo"])["stars_hourly"].cumsum()
    # convert ts to ISO
    g["ts_hour"] = g["ts_hour"].dt.tz_convert(timezone.utc).dt.strftime("%Y-%m-%dT%H:00:00Z")
    return g

def main():
    parser = argparse.ArgumentParser(description="Fetch stargazer timestamps and build hourly star time series.")
    parser.add_argument("--input_csv", default="data/processed/github_repos_from_hn.csv", help="Repo list CSV (owner,repo)")
    parser.add_argument("--out_csv", default=OUT_TS_CSV, help="Output CSV for timeseries (append/resume-safe)")
    parser.add_argument("--max_repos", type=int, default=0, help="Optional cap for quick tests")
    parser.add_argument("--force_repo", type=str, default="", help="Process only this owner/repo (format owner/repo)")
    args = parser.parse_args()

    ensure_dirs()
    load_dotenv()
    token = os.environ.get("GH_TOKEN","")

    if not os.path.exists(args.input_csv):
        print(f"Missing input CSV: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    src = pd.read_csv(args.input_csv)
    src = src[(src["owner"].notna()) & (src["repo"].notna())]
    src["owner"] = src["owner"].astype(str).str.strip().str.lower()
    src["repo"]  = src["repo"].astype(str).str.strip().str.lower()
    pairs = src.drop_duplicates(subset=["owner","repo"])[["owner","repo"]].values.tolist()

    if args.force_repo:
        want = args.force_repo.strip().lower()
        pairs = [p for p in pairs if f"{p[0]}/{p[1]}" == want]

    done_pairs = set()
    if os.path.exists(args.out_csv):
        prev = pd.read_csv(args.out_csv)
        if not prev.empty:
            # consider a pair "done" if it has at least one row in out_csv
            done_pairs = set((prev["owner"].str.lower() + "/" + prev["repo"].str.lower()).unique().tolist())

    total_events = 0
    processed = 0
    appended_rows = 0

    out_chunks = []

    for i, (owner, repo) in enumerate(pairs, start=1):
        key = f"{owner}/{repo}"
        if key in done_pairs:
            continue
        if args.max_repos and processed >= args.max_repos:
            break

        raw_jsonl = f"data/raw/stargazers_{owner}_{repo}.jsonl"
        # If you want a clean re-run for one repo, delete the jsonl file manually.

        # fetch events
        got = fetch_stargazers(owner, repo, token, raw_jsonl)
        total_events += got

        # build hourly TS
        df = build_hourly_timeseries(owner, repo, raw_jsonl)
        if not df.empty:
            out_chunks.append(df)
            appended_rows += len(df)

        processed += 1
        time.sleep(0.2)

    if out_chunks:
        new_df = pd.concat(out_chunks, ignore_index=True)
        if os.path.exists(args.out_csv):
            old_df = pd.read_csv(args.out_csv)
            all_df = pd.concat([old_df, new_df], ignore_index=True)
            all_df = all_df.drop_duplicates(subset=["owner","repo","ts_hour"])
            all_df.to_csv(args.out_csv, index=False)
        else:
            new_df.to_csv(args.out_csv, index=False)

    # summary
    lines = []
    lines.append(f"Repos processed (this run): {processed}")
    lines.append(f"Star events fetched (this run): {total_events}")
    lines.append(f"Rows appended to timeseries (this run): {appended_rows}")
    lines.append(f"Timeseries CSV: {args.out_csv}")
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.out_csv} and {SUMMARY}")

if __name__ == "__main__":
    main()

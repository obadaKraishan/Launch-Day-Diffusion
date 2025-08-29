#!/usr/bin/env python3
"""
03_github_repo_metadata.py
Enrich GitHub repos with static metadata: owner type, created_at, license, topics,
readme length, releases presence, stars_now, forks, open_issues, watchers/subscribers.
Saves:
  - data/raw/github_repos_metadata.jsonl   (one JSON blob per repo with raw fields)
  - data/processed/github_repos_metadata.csv
  - outputs/summaries/03_github_repo_metadata_summary.txt
Usage:
  python src/03_github_repo_metadata.py --input_csv data/processed/github_repos_from_hn.csv
"""
import os, sys, json, time, argparse
from typing import Tuple, Dict, Any
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

RAW_PATH = "data/raw/github_repos_metadata.jsonl"
OUT_CSV  = "data/processed/github_repos_metadata.csv"
SUMMARY  = "outputs/summaries/03_github_repo_metadata_summary.txt"

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/summaries", exist_ok=True)

def gh_headers(token: str = "") -> Dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "icwsm-hn-github-metadata"
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def backoff_sleep(attempt: int):
    time.sleep(min(60, 2 ** attempt))

def gh_get(url: str, token: str, params: Dict[str, Any] = None, accept: str = None, allow_404=False) -> Tuple[int, Any, Dict[str,str]]:
    headers = gh_headers(token)
    if accept:
        headers["Accept"] = accept
    attempt = 0
    while True:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 200:
            return 200, r.json(), r.headers
        if allow_404 and r.status_code == 404:
            return 404, None, r.headers
        if r.status_code in (403, 429, 502, 503):
            attempt += 1
            backoff_sleep(attempt)
            continue
        # other failures: return status and text
        try:
            return r.status_code, r.json(), r.headers
        except Exception:
            return r.status_code, {"error": r.text[:200]}, r.headers

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def main():
    parser = argparse.ArgumentParser(description="Fetch GitHub repo metadata for repos extracted from HN.")
    parser.add_argument("--input_csv", default="data/processed/github_repos_from_hn.csv", help="Repo list CSV with columns owner,repo")
    parser.add_argument("--raw_jsonl", default=RAW_PATH, help="Output JSONL path for raw blobs")
    parser.add_argument("--out_csv", default=OUT_CSV, help="Output CSV path for tidy metadata")
    parser.add_argument("--force", action="store_true", help="Ignore existing out_csv and process all")
    parser.add_argument("--max_repos", type=int, default=0, help="Optional cap for debugging")
    args = parser.parse_args()

    ensure_dirs()
    load_dotenv()
    token = os.environ.get("GH_TOKEN", "")

    if not os.path.exists(args.input_csv):
        print(f"Missing input: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    src = pd.read_csv(args.input_csv)
    src = src[(src["owner"].notna()) & (src["repo"].notna())]
    src["owner"] = src["owner"].astype(str).str.strip()
    src["repo"]  = src["repo"].astype(str).str.strip()

    # resume support: skip already processed repos in out_csv (unless --force)
    done = set()
    if os.path.exists(args.out_csv) and not args.force:
        prev = pd.read_csv(args.out_csv)
        done = set((prev["owner"].str.lower() + "/" + prev["repo"].str.lower()).tolist())

    rows = []
    processed = 0
    errors = 0

    for owner, repo in src.drop_duplicates(subset=["owner","repo"])[["owner","repo"]].itertuples(index=False):
        key = f"{owner.lower()}/{repo.lower()}"
        if key in done:
            continue
        if args.max_repos and processed >= args.max_repos:
            break

        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        status, repo_json, headers = gh_get(repo_url, token)
        if status != 200:
            errors += 1
            # still write a raw line with the error for traceability
            with open(args.raw_jsonl, "a", encoding="utf-8") as fw:
                fw.write(json.dumps({"owner": owner, "repo": repo, "error": repo_json, "ts": datetime.utcnow().isoformat()+"Z"}) + "\n")
            continue

        # topics
        topics_url = f"https://api.github.com/repos/{owner}/{repo}/topics"
        s2, topics_json, _ = gh_get(topics_url, token)
        topics = topics_json.get("names", []) if s2 == 200 and isinstance(topics_json, dict) else []

        # readme (length only)
        readme_len = 0
        readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        s3, readme_json, _ = gh_get(readme_url, token, allow_404=True)
        if s3 == 200 and isinstance(readme_json, dict):
            # either 'size' or base64 'content'
            if "size" in readme_json:
                readme_len = safe_int(readme_json.get("size", 0))
            elif "content" in readme_json and isinstance(readme_json["content"], str):
                readme_len = len(readme_json["content"])

        # releases presence (peek 1)
        rel_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        s4, rel_json, _ = gh_get(rel_url, token, params={"per_page": 1})
        has_releases = (s4 == 200 and isinstance(rel_json, list) and len(rel_json) > 0)

        owner_type = (repo_json.get("owner") or {}).get("type", "")
        license_spdx = (repo_json.get("license") or {}).get("spdx_id", "") if repo_json.get("license") else ""
        watchers = repo_json.get("subscribers_count", repo_json.get("watchers_count", 0))

        row = {
            "owner": owner,
            "repo": repo,
            "owner_type": owner_type,
            "created_at": repo_json.get("created_at", ""),
            "pushed_at": repo_json.get("pushed_at", ""),
            "license_spdx": license_spdx,
            "default_branch": repo_json.get("default_branch", ""),
            "stars_now": repo_json.get("stargazers_count", 0),
            "forks": repo_json.get("forks_count", 0),
            "open_issues": repo_json.get("open_issues_count", 0),
            "watchers": watchers,
            "topics_json": json.dumps(topics, ensure_ascii=False),
            "readme_len": readme_len,
            "has_releases": has_releases
        }
        rows.append(row)

        # write raw blob for provenance
        raw_blob = {
            "owner": owner, "repo": repo,
            "repo_json": repo_json,
            "topics": topics,
            "readme_len": readme_len,
            "has_releases": has_releases,
            "ts": datetime.utcnow().isoformat()+"Z"
        }
        with open(args.raw_jsonl, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(raw_blob, ensure_ascii=False) + "\n")

        processed += 1

        # friendly pacing on API
        time.sleep(0.2)

    # append to existing CSV or create new
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        pass
    elif os.path.exists(args.out_csv) and not args.force:
        prev = pd.read_csv(args.out_csv)
        all_df = pd.concat([prev, out_df], ignore_index=True)
        all_df = all_df.drop_duplicates(subset=["owner","repo"])
        all_df.to_csv(args.out_csv, index=False)
    else:
        out_df.to_csv(args.out_csv, index=False)

    # summary
    lines = []
    uniq = len(out_df.drop_duplicates(subset=["owner","repo"])) if not out_df.empty else 0
    lines.append(f"Processed repos (this run): {uniq}")
    lines.append(f"Errors (non-200): {errors}")
    lines.append(f"Raw JSONL: {args.raw_jsonl}")
    lines.append(f"CSV out: {args.out_csv}")
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.out_csv} and {SUMMARY}")

if __name__ == "__main__":
    main()

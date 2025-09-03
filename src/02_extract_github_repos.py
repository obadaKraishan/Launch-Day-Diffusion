"""
02_extract_github_repos.py
Parse and validate GitHub owner/repo from HN post URLs.
Saves: CSV (mapping), TXT (summary)
Author: <OBADA KRAISHAN>
"""
import argparse, os, re, sys, csv, time
import pandas as pd
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv

GITHUB_API = "https://api.github.com/repos/{owner}/{repo}"

REPO_RE = re.compile(r"github\.com/([^/]+)/([^/#?]+)", re.IGNORECASE)

def parse_args():
    ap = argparse.ArgumentParser(description="Extract and validate GitHub repos from hn_posts.csv")
    ap.add_argument("--input_csv", type=str, default="data/processed/hn_posts.csv", help="Path to hn_posts.csv")
    ap.add_argument("--output_csv", type=str, default="data/processed/github_repos_from_hn.csv", help="Output CSV path")
    ap.add_argument("--validate", action="store_true", help="Validate repos exist via GitHub API")
    return ap.parse_args()

def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/summaries", exist_ok=True)

def classify_path(url):
    # returns 'repo', 'issues', 'pull', 'blob', 'tree', or 'other'
    try:
        path = urlparse(url).path.lower()
    except Exception:
        return "other"
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        return "other"
    if "issues" in parts:
        return "issues"
    if "pull" in parts or "pulls" in parts:
        return "pull"
    if "blob" in parts:
        return "blob"
    if "tree" in parts:
        return "tree"
    return "repo"

def extract_owner_repo(url):
    if not url:
        return None, None
    m = REPO_RE.search(url)
    if not m:
        return None, None
    owner = m.group(1).strip()
    repo = m.group(2).strip()
    # strip .git and trailing punctuation
    repo = re.sub(r"\.git$", "", repo)
    repo = repo.rstrip(").,;:!?\"'")
    return owner, repo

def gh_validate(owner, repo, token=None):
    url = GITHUB_API.format(owner=owner, repo=repo)
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code == 200:
        return True, "ok"
    else:
        try:
            msg = r.json().get("message","")
        except Exception:
            msg = r.text[:120]
        return False, f"{r.status_code}: {msg}"

def main():
    args = parse_args()
    ensure_dirs()
    load_dotenv()  # loads GH_TOKEN if present

    token = os.environ.get("GH_TOKEN","")

    if not os.path.exists(args.input_csv):
        print(f"Input CSV not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input_csv)
    rows = []
    for _, r in df.iterrows():
        url = r.get("resolved_url") if isinstance(r.get("resolved_url"), str) and r.get("resolved_url") else r.get("url")
        path_type = classify_path(url or "")
        owner, repo = extract_owner_repo(url or "")
        valid = None
        reason = ""
        if owner and repo and args.validate:
            ok, reason = gh_validate(owner, repo, token=token)
            valid = ok
        elif owner and repo:
            valid = None  # unknown (not checked)
        rows.append({
            "hn_id": int(r.get("hn_id")) if pd.notnull(r.get("hn_id")) else None,
            "title": r.get("title",""),
            "original_url": r.get("url",""),
            "resolved_url": r.get("resolved_url",""),
            "owner": (owner or "").lower(),
            "repo": (repo or "").lower(),
            "path_type": path_type,
            "valid": valid,
            "valid_reason": reason
        })

    out_df = pd.DataFrame(rows)
    # drop rows without owner/repo
    out_df = out_df[(out_df["owner"]!="") & (out_df["repo"]!="")]
    out_df = out_df.drop_duplicates(subset=["owner","repo","hn_id"])

    out_df.to_csv(args.output_csv, index=False)

    unique_repos = out_df.drop_duplicates(subset=["owner","repo"]).shape[0]
    invalids = out_df[(out_df["valid"]==False)].shape[0]

    lines = []
    lines.append(f"Unique repos: {unique_repos}")
    lines.append(f"Rows (owner/repo mapped): {out_df.shape[0]}")
    lines.append(f"Invalid repos (validated & 404/err): {invalids}")
    sample = out_df.head(10)[["owner","repo","path_type","valid"]].to_dict(orient="records")
    lines.append("Sample (first 10):")
    for s in sample:
        lines.append(f"- {s['owner']}/{s['repo']} [{s['path_type']}] valid={s['valid']}")

    with open("outputs/summaries/02_extract_github_repos_summary.txt","w",encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.output_csv} and outputs/summaries/02_extract_github_repos_summary.txt")

if __name__ == "__main__":
    main()

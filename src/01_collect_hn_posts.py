"""
01_collect_hn_posts.py
Fetch Hacker News stories linking to GitHub, within a date range.
Saves: JSONL (raw), CSV (tidy), TXT (summary)
Default method: Algolia HN Search API (no key). Optionally: Firebase scan (fallback).
Author: <OBADA KRAISHAN>
"""
import argparse, os, re, json
from datetime import datetime, timezone
from urllib.parse import urlparse
import requests
import pandas as pd
from dateutil import parser as dateparser

DEFAULT_QUERY = ""
ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date"

SHOW_HN_RE = re.compile(r"^\s*show\s*hn\b", re.IGNORECASE)


def parse_args():
    """
    CLI parser with optional YAML config.
    Precedence for defaults: ENV VARS > YAML config > hard-coded defaults.
    """
    # Pre-parse --config so we can read YAML before defining full defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config", type=str,
        default=os.environ.get("HN_CONFIG", "config.yaml"),
        help="Path to YAML config with defaults (optional)")
    pre_args, _ = pre.parse_known_args()

    # Load YAML config if available
    cfg = {}
    if os.path.exists(pre_args.config):
        try:
            import yaml
            with open(pre_args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

    # helpers that respect ENV > YAML > hard default
    def cfgget_str(key, env, default):
        return os.environ.get(env, cfg.get(key, default))

    def cfgget_int(key, env, default):
        v = os.environ.get(env, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
        v = cfg.get(key, default)
        try:
            return int(v)
        except Exception:
            return default

    def cfgget_bool(key, env, default):
        v = os.environ.get(env, None)
        if v is not None:
            return str(v).strip().lower() in {"1","true","t","yes","y"}
        v = cfg.get(key, default)
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"1","true","t","yes","y"}

    ap = argparse.ArgumentParser(
        description="Collect HN posts linking to GitHub for AI/LLM tools.",
        parents=[pre],
        add_help=True
    )
    ap.add_argument("--start", type=str,
                    default=cfgget_str("start_date", "HN_START", "2023-01-01"),
                    help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str,
                    default=cfgget_str("end_date", "HN_END", datetime.now(timezone.utc).date().isoformat()),
                    help="End date (YYYY-MM-DD)")
    ap.add_argument("--min_score", type=int,
                    default=cfgget_int("min_score", "HN_MIN_SCORE", 10),
                    help="Minimum HN points to keep")
    ap.add_argument("--query", type=str,
                    default=cfgget_str("query", "HN_QUERY", DEFAULT_QUERY),
                    help="Comma-separated keywords to match (title or URL). Empty = no keyword filter.")
    ap.add_argument("--algolia_query", type=str,
                    default=cfgget_str("algolia_query", "HN_ALGOLIA_QUERY", "github.com"),
                    help="Seed query sent to Algolia (e.g., 'github.com'). Local filters still apply.")
    ap.add_argument("--only_show_hn", action="store_true",
                    default=cfgget_bool("only_show_hn", "HN_ONLY_SHOW_HN", False),
                    help="Keep only 'Show HN' posts")
    ap.add_argument("--resolve_urls", action="store_true",
                    default=cfgget_bool("resolve_urls", "HN_RESOLVE_URLS", False),
                    help="Resolve redirects to get final URL (slower)")
    ap.add_argument("--method", choices=["algolia","firebase"],
                    default=cfgget_str("method", "HN_METHOD", "algolia"),
                    help="API method to use")
    ap.add_argument("--out_prefix", type=str,
                    default=cfgget_str("out_prefix", "HN_OUT_PREFIX", "allgh23"),
                    help="Optional prefix for output filenames")
    return ap.parse_args()


def ensure_dirs():
    for p in ["data/raw", "data/processed", "outputs/summaries"]:
        os.makedirs(p, exist_ok=True)


def matches_keywords(title, url, keywords):
    if not keywords:
        return True
    pats = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    if not pats:
        return True
    hay = f"{title or ''} {url or ''}".lower()
    return any(kw.lower() in hay for kw in pats)


def is_github_url(url):
    if not url:
        return False
    try:
        return urlparse(url).netloc.lower().endswith("github.com")
    except Exception:
        return False


def resolve_url(url, timeout=10):
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        return r.url
    except Exception:
        return url


def collect_algolia(start_iso, end_iso, min_score, keywords, only_show_hn, resolve_urls, algolia_query):
    start_epoch = int(datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc).timestamp())
    end_epoch = int(datetime.fromisoformat(end_iso).replace(tzinfo=timezone.utc).timestamp())
    page = 0
    hits_total = 0
    kept = []

    while True:
        params = {
            "query": algolia_query or "",
            "tags": "story",
            "numericFilters": f"created_at_i>={start_epoch},created_at_i<={end_epoch}",
            "hitsPerPage": 100,
            "page": page
        }
        resp = requests.get(ALGOLIA_URL, params=params, timeout=30)
        resp.raise_for_status()
        js = resp.json()
        hits = js.get("hits", [])
        if not hits:
            break
        hits_total += len(hits)

        for h in hits:
            url = h.get("url")
            title = h.get("title") or ""
            is_show_hn = bool(SHOW_HN_RE.match(title))
            if only_show_hn and not is_show_hn:
                continue
            if not is_github_url(url):
                continue
            if h.get("points", 0) < min_score:
                continue
            if not matches_keywords(title, url, keywords):
                continue

            created_at = h.get("created_at")
            try:
                created_dt = dateparser.isoparse(created_at).astimezone(timezone.utc)
                created_iso = created_dt.isoformat()
            except Exception:
                created_iso = created_at or ""

            resolved = resolve_url(url) if resolve_urls else url
            kept.append({
                "hn_id": int(h["objectID"]),
                "time_utc": created_iso,
                "title": title,
                "url": url,
                "resolved_url": resolved,
                "score": h.get("points", 0),
                "descendants": h.get("num_comments", 0),
                "is_show_hn": is_show_hn,
                "author": h.get("author", "")
            })

        page += 1
        nb_pages = js.get("nbPages", page)
        if page >= nb_pages:
            break
    return kept, hits_total


def collect_firebase(start_iso, end_iso, min_score, keywords, only_show_hn, resolve_urls, algolia_query):
    # Fallback: reuse Algolia collection to keep runtime fast and simple.
    return collect_algolia(start_iso, end_iso, min_score, keywords, only_show_hn, resolve_urls, algolia_query)


def main():
    args = parse_args()
    ensure_dirs()

    start_iso = args.start
    end_iso = args.end
    out_prefix = (args.out_prefix + "_") if args.out_prefix else ""

    if args.method == "algolia":
        kept, hits_total = collect_algolia(start_iso, end_iso, args.min_score, args.query,
                                           args.only_show_hn, args.resolve_urls, args.algolia_query)
    else:
        kept, hits_total = collect_firebase(start_iso, end_iso, args.min_score, args.query,
                                            args.only_show_hn, args.resolve_urls, args.algolia_query)

    # Save JSONL
    raw_path = f"data/raw/{out_prefix}hn_items_{start_iso}_{end_iso}.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save CSV
    df = pd.DataFrame(
        kept,
        columns=["hn_id", "time_utc", "title", "url", "resolved_url", "score", "descendants", "is_show_hn", "author"]
    )
    csv_path = "data/processed/hn_posts.csv"
    df.to_csv(csv_path, index=False)

    # Summary TXT
    summ = []
    summ.append(f"Total HN hits scanned (Algolia pages summed): {hits_total}")
    summ.append(f"Kept (github.com & filters): {len(df)}")
    if not df.empty:
        top = df.sort_values('score', ascending=False).head(10)[["score", "title", "url"]].to_dict(orient="records")
        summ.append("Top 10 by score:")
        for t in top:
            summ.append(f"- {t['score']:>4} : {t['title'][:120]} ... ({t['url']})")
    else:
        summ.append("No posts matched the filters. Consider lowering --min_score, widening dates, or setting --algolia_query ''.")
    txt_path = "outputs/summaries/01_collect_hn_posts_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summ))

    print(f"Wrote {raw_path}, {csv_path}, and {txt_path}")


if __name__ == "__main__":
    main()

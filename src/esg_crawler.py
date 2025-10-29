#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
esg_crawler_brave.py
Brave Search API–driven ESG / Sustainability PDF crawler.
"""
import os, re, csv, time, argparse, hashlib
from urllib.parse import urlparse
import requests

def sha256_bytes(b):
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def read_sp500_csv(path):
    tickers = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            t = (row.get("ticker") or "").strip()
            if t:
                tickers.append(t)
    return tickers

def read_company_map(path):
    mp = {}
    if not path or not os.path.exists(path):
        return mp
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            t = (r.get("ticker") or "").strip()
            c = (r.get("company") or "").strip()
            if t and c:
                mp[t] = c
    return mp

def load_keywords(root):
    try:
        import yaml
        p = os.path.join(root, "config", "keywords.yaml")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    return {"report_keywords": ["sustainability report","ESG report","corporate responsibility report","CSR report"]}

def make_queries(ticker, company_name, years, report_keywords):
    name = company_name or ticker
    base_kw = " OR ".join([f'\"{kw}\"' for kw in report_keywords[:4]])
    q_list = []
    for y in years:
        q = f'{name} ({base_kw}) "{y}" filetype:pdf'
        q_list.append(q)
    return q_list

def brave_search(q, key, max_results=10):
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"X-Subscription-Token": key}
    params = {"q": q, "count": max_results, "search_lang": "en", "spellcheck": "0"}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("web", {}).get("results", []):
        link = item.get("url")
        title = item.get("title","")
        if link:
            results.append((link, title))
    return results

def is_pdf_url(url):
    return urlparse(url).path.lower().endswith(".pdf")

def content_type_pdf(headers):
    ct = (headers.get("Content-Type") or "").lower()
    return "application/pdf" in ct or ct.endswith("/pdf")

def download_pdf(url, timeout=45, headers=None):
    r = requests.get(url, timeout=timeout, headers=headers or {"User-Agent":"Mozilla/5.0 (ESGBot/1.0)"})
    if r.status_code == 200 and content_type_pdf(r.headers):
        return r.content, r.headers
    return None, r.headers

def safe_filename(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--start-year", type=int, default=2017)
    ap.add_argument("--end-year", type=int, default=2018)
    ap.add_argument("--brave-key", required=True)
    ap.add_argument("--company-map", default=None)
    ap.add_argument("--max-results", type=int, default=8)
    ap.add_argument("--throttle-sec", type=float, default=1.0)
    args = ap.parse_args()

    root = args.root
    sp500_csv = os.path.join(root, "data", "sp500_2017-01-27.csv")
    manifest_path = os.path.join(root, "data", "metadata", "esg_manifest.csv")
    raw_root = os.path.join(root, "data", "raw")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    os.makedirs(raw_root, exist_ok=True)

    years = list(range(args.start_year, args.end_year+1))
    kw = load_keywords(root)
    report_keywords = kw.get("report_keywords", [])
    tickers = read_sp500_csv(sp500_csv)
    company_map = read_company_map(args.company_map)

    write_header = not os.path.exists(manifest_path)
    mf = open(manifest_path, "a", newline="", encoding="utf-8")
    w = csv.writer(mf)
    if write_header:
        w.writerow(["ticker","year","url","title","source","mime","bytes","sha256","status"])

    for tkr in tickers:
        name = company_map.get(tkr, None)
        queries = make_queries(tkr, name, years, report_keywords)
        for q in queries:
            try:
                results = brave_search(q, args.brave_key, max_results=args.max_results)
            except Exception:
                time.sleep(args.throttle_sec * 2)  # typo here would raise; we'll keep it correct below
                continue
            for (link, title) in results:
                if not is_pdf_url(link):
                    continue
                year_in_q = None
                for yy in years:
                    if str(yy) in q:
                        year_in_q = yy
                        break
                if year_in_q is None:
                    continue
                try:
                    blob, headers = download_pdf(link)
                    if blob:
                        digest = sha256_bytes(blob)
                        out_dir = os.path.join(raw_root, tkr, str(year_in_q))
                        os.makedirs(out_dir, exist_ok=True)
                        fname = safe_filename(f"{tkr}_{year_in_q}_{digest[:8]}.pdf")
                        out_path = os.path.join(out_dir, fname)
                        with open(out_path, "wb") as f:
                            f.write(blob)
                        w.writerow([tkr, year_in_q, link, title, "brave", "application/pdf", len(blob), digest, "downloaded"])
                    else:
                        w.writerow([tkr, year_in_q, link, title, "brave", (headers or {}).get("Content-Type",""), "", "", "skipped_non_pdf"])
                except Exception as e:
                    w.writerow([tkr, year_in_q, link, title, "brave", "", "", "", f"error:{type(e).__name__}"])
                mf.flush()
                time.sleep(args.throttle_sec)
        time.sleep(args.throttle_sec * 2)

    mf.close()
    print("Done. Manifest at:", manifest_path)

if __name__ == "__main__":
    main()

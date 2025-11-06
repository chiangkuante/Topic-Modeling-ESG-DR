#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG Brave Search crawler with pagination, retries, dedupe, and manifest resume support.
"""
import argparse
import csv
import hashlib
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 載入環境變數
from dotenv import load_dotenv
load_dotenv()

PDF_SIGNATURE = b"%PDF"
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; ESGCrawler/1.1; +https://example.com/contact)"
DEFAULT_REPORT_KEYWORDS = [
    "sustainability report",
    "ESG report",
    "corporate responsibility report",
    "CSR report",
    "non-financial report",
    "sustainability accounting",
]
AGGREGATOR_BLOCKLIST = ["issuu.com", "scribd.com", "slideshare.net"]
MANIFEST_COLUMNS = [
    "timestamp",
    "ticker",
    "year",
    "query",
    "url",
    "final_url",
    "title",
    "source",
    "mime",
    "bytes",
    "sha256",
    "status",
    "status_detail",
    "http_status",
    "error_reason",
    "robots_allowed",
    "content_disposition",
    "saved_path",
]


def sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def read_sp500_csv(path: str) -> List[str]:
    tickers: List[str] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = (row.get("ticker") or "").strip()
            if ticker:
                tickers.append(ticker)
    return tickers


def read_company_map(path: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return mapping
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = (row.get("ticker") or "").strip()
            company = (row.get("company") or "").strip()
            if ticker and company:
                mapping[ticker] = company
    return mapping


def load_keywords(root: str) -> Dict[str, Sequence[str]]:
    try:
        import yaml  # type: ignore

        path = os.path.join(root, "config", "keywords.yaml")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {"report_keywords": DEFAULT_REPORT_KEYWORDS}


def ensure_manifest_header(manifest_path: str) -> None:
    if not os.path.exists(manifest_path):
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
        return

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            header = []
    if header == MANIFEST_COLUMNS:
        return

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            upgraded = {key: row.get(key, "") for key in MANIFEST_COLUMNS}
            if not upgraded.get("timestamp"):
                upgraded["timestamp"] = ""
            if not upgraded.get("robots_allowed"):
                upgraded["robots_allowed"] = ""
            writer.writerow(upgraded)


def load_manifest_index(manifest_path: str) -> Tuple[Set[str], Set[str]]:
    seen_urls: Set[str] = set()
    seen_hashes: Set[str] = set()
    if not os.path.exists(manifest_path):
        return seen_urls, seen_hashes
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = (row.get("url") or "").strip()
            digest = (row.get("sha256") or "").strip()
            if url:
                seen_urls.add(url.lower())
            if digest:
                seen_hashes.add(digest)
    return seen_urls, seen_hashes


def is_pdf_like_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith(".pdf"):
        return True
    return ".pdf" in path


def is_pdf_mime(headers: requests.structures.CaseInsensitiveDict) -> bool:
    content_type = (headers.get("Content-Type") or "").lower()
    if "application/pdf" in content_type or content_type.endswith("/pdf"):
        return True
    if "application/octet-stream" in content_type:
        return True
    return False


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def get_clean_domain(url: str) -> str:
    """提取並清理網域，例如 'https://www.google.com/path' -> 'google'"""
    try:
        netloc = urlparse(url).netloc
        if not netloc:
            return ""

        parts = netloc.split('.')
        # 移除 'www' 並取最後兩個主要部分 (例如 'google.com' -> 'google')
        # 對於 'co.uk' 這種會有點問題，但對於 S&P 500 公司來說，這個簡易邏輯足夠了
        if len(parts) > 1:
            # 返回 'google' 而不是 'google.com'
            return parts[-2].lower()
        return parts[0].lower()
    except Exception:
        return ""


def is_domain_related(company_name: str, url: str) -> bool:
    """檢查 URL 網域是否與公司名稱相關"""
    if not company_name:
        return True  # 如果沒有公司名稱可比對，只能放行

    clean_domain = get_clean_domain(url)
    if not clean_domain:
        return False  # 無法解析網域，視為不相關

    # 特殊情況: 如果 company_name 是短的 ticker (全大寫且 <=6 字元)
    # 則只做非常寬鬆的檢查或直接放行，因為 ticker 通常與網域無法直接匹配
    if company_name.isupper() and len(company_name) <= 6:
        # 對於 ticker，只檢查是否完全匹配或是網域的一部分
        ticker_lower = company_name.lower()
        if ticker_lower == clean_domain or ticker_lower in clean_domain or clean_domain in ticker_lower:
            return True
        # 如果 ticker 和 domain 完全不匹配，則放行 (因為無法從 ticker 推斷公司全名)
        # 這種情況應該依賴 PDF 內容的後續驗證
        return True

    # 清理公司名稱 (移除 Inc, Corp, . 等)
    clean_name = re.sub(r"[,.(](inc|corp|ltd|plc|company|services|group|holdings)[).,]*", "", company_name.lower(), flags=re.IGNORECASE)
    clean_name = re.sub(r"[^a-z0-9\s]", "", clean_name).strip()

    # 移除空格和連字符，方便比對 (如 "JPMorgan Chase" -> "jpmorganchase")
    clean_name_nospace = clean_name.replace(" ", "")

    # 例如 clean_name = "alphabet" (來自 "Alphabet Inc.")
    # clean_domain = "abc" (來自 "abc.xyz") -> 需要更寬鬆的比對
    # clean_domain = "google" (來自 "google.com") -> 相關

    # 策略 1: 完整名稱比對 (無空格版本)
    if clean_name_nospace in clean_domain or clean_domain in clean_name_nospace:
        return True

    # 策略 2: 逐字比對 (每個單詞至少3個字元)
    for name_part in clean_name.split():
        if len(name_part) >= 3:
            if name_part in clean_domain or clean_domain in name_part:
                return True

    # 策略 3: 首字母縮寫比對 (如 "International Business Machines" -> "ibm")
    # 只適用於多單詞公司名稱
    name_words = [w for w in clean_name.split() if len(w) >= 2]
    if len(name_words) >= 2:
        initials = "".join(w[0] for w in name_words)
        if initials == clean_domain or (len(initials) >= 3 and initials in clean_domain):
            return True

    # 策略 4: 部分匹配 (至少4個連續字元匹配)
    # 用於處理縮寫或簡化的網域名稱
    if len(clean_domain) >= 4 and len(clean_name_nospace) >= 4:
        # 檢查網域是否是公司名稱的連續子字串
        if clean_domain in clean_name_nospace:
            return True
        # 檢查公司名稱是否包含網域的大部分
        if len(clean_domain) >= 5:
            # 使用滑動窗口檢查
            for i in range(len(clean_name_nospace) - len(clean_domain) + 1):
                substring = clean_name_nospace[i:i+len(clean_domain)]
                # 計算相似度 (簡單的字元匹配)
                matches = sum(1 for a, b in zip(substring, clean_domain) if a == b)
                if matches >= len(clean_domain) * 0.7:  # 70% 匹配度
                    return True

    return False


class PDFLinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[Tuple[str, str]] = []  # (href, link_text)
        self._current_href: Optional[str] = None
        self._current_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        attrs_dict = {key.lower(): (val or "") for key, val in attrs}

        if tag_lower in ("a", "area"):
            href = attrs_dict.get("href")
            if href:
                # 開始捕獲新連結
                self._current_href = href
                self._current_text = []
        elif tag_lower == "meta":
            http_equiv = attrs_dict.get("http-equiv", "")
            if http_equiv.lower() == "refresh":
                content = attrs_dict.get("content", "")
                match = re.search(r"url=(.+)", content, re.IGNORECASE)
                if match:
                    url = match.group(1).strip()
                    self.links.append((url, "meta_refresh"))  # meta refresh 沒有連結文字

    def handle_data(self, data: str) -> None:
        # 如果正在捕獲連結，就儲存文字
        if self._current_href and data.strip():
            self._current_text.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("a", "area"):
            # 結束捕獲
            if self._current_href:
                link_text = " ".join(self._current_text)
                self.links.append((self._current_href, link_text))
            self._current_href = None
            self._current_text = []

    def get_scored_pdf_links(self, base_url: str, keywords: List[str]) -> List[Tuple[int, str]]:
        """
        對所有提取的連結進行評分並返回 (score, absolute_url) 列表
        """
        scored_links: List[Tuple[int, str]] = []
        keywords_lower = [k.lower() for k in keywords if k]

        for href, text in self.links:
            try:
                absolute_url = urljoin(base_url, href.strip())
            except Exception:
                continue  # 無效的 href

            # 只考慮 PDF 類型的連結
            if not is_pdf_like_url(absolute_url):
                continue

            score = 0
            url_lower = absolute_url.lower()
            text_lower = text.lower()

            # 根據 URL 和連結文字中的關鍵字評分
            # "ESG report" 有最高優先權
            if "esg" in url_lower and "report" in url_lower:
                score += 20  # ESG report 同時出現在 URL 中
            if "esg" in text_lower and "report" in text_lower:
                score += 20  # ESG report 同時出現在連結文字中

            # 其他關鍵字評分
            for keyword in keywords_lower:
                if keyword in url_lower:
                    score += 5
                if keyword in text_lower:
                    score += 5

            if "report" in url_lower or "report" in text_lower:
                score += 3
            if "sustainability" in url_lower or "sustainability" in text_lower:
                score += 3
            if "esg" in url_lower or "esg" in text_lower:
                score += 3

            # 降低不太相關文件的分數
            if "annual" in url_lower or "annual" in text_lower:
                score -= 1  # 可能是年報
            if "financial" in url_lower or "financial" in text_lower:
                score -= 3  # 可能是財報
            if "proxy" in url_lower or "proxy" in text_lower:
                score -= 3  # 代理投票說明書
            if "earnings" in url_lower or "earnings" in text_lower:
                score -= 3  # 財報
            if "10-k" in url_lower or "10-q" in url_lower:
                score -= 5  # SEC 文件

            if score > 0:
                scored_links.append((score, absolute_url))

        # 按分數從高到低排序
        scored_links.sort(key=lambda x: x[0], reverse=True)
        return scored_links


def extract_pdf_link(html: str, base_url: str) -> Optional[str]:
    """向後相容的函數，返回第一個 PDF 連結"""
    parser = PDFLinkExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    for href, _text in parser.links:
        absolute = urljoin(base_url, href.strip())
        if is_pdf_like_url(absolute):
            return absolute
    return None


class AdaptiveThrottle:
    def __init__(self, base_delay: float, max_delay: float) -> None:
        self.base_delay = max(base_delay, 0.1)
        self.current_delay = self.base_delay
        self.max_delay = max(max_delay, self.base_delay)

    def wait(self) -> None:
        time.sleep(self.current_delay)

    def success(self) -> None:
        if self.current_delay > self.base_delay:
            self.current_delay = max(self.base_delay, self.current_delay * 0.9)

    def backoff(self) -> None:
        self.current_delay = min(self.max_delay, self.current_delay * 1.5)


class RobotsCache:
    def __init__(self, session: requests.Session, user_agent: str, throttle: AdaptiveThrottle, respect: bool) -> None:
        self.session = session
        self.user_agent = user_agent
        self.throttle = throttle
        self.respect = respect
        self.cache: Dict[str, Optional[robotparser.RobotFileParser]] = {}

    def allowed(self, url: str) -> Tuple[bool, Optional[bool]]:
        if not self.respect:
            return True, None
        parsed = urlparse(url)
        key = f"{parsed.scheme}://{parsed.netloc}"
        if key not in self.cache:
            robots_url = f"{key}/robots.txt"
            try:
                self.throttle.wait()
                resp = self.session.get(
                    robots_url,
                    headers={"User-Agent": self.user_agent},
                    timeout=20,
                )
                if resp.status_code == 429:
                    self.throttle.backoff()
                resp.raise_for_status()
                parser = robotparser.RobotFileParser()
                parser.parse(resp.text.splitlines())
                self.cache[key] = parser
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status in {401, 403}:
                    self.cache[key] = robotparser.RobotFileParser()
                    self.cache[key].parse([])
                else:
                    self.cache[key] = None
            except requests.exceptions.RequestException:
                self.cache[key] = None
        parser = self.cache.get(key)
        if parser is None:
            return True, None
        allowed = parser.can_fetch(self.user_agent, url)
        return allowed, allowed


@dataclass
class QueryTask:
    ticker: str
    year: int
    query: str


class BraveSearchClient:
    def __init__(self, session: requests.Session, api_key: str, throttle: AdaptiveThrottle, user_agent: str) -> None:
        self.session = session
        self.api_key = api_key
        self.throttle = throttle
        self.user_agent = user_agent
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, max_per_page: int, max_total: int) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        offset = 0
        seen_urls: Set[str] = set()
        while len(results) < max_total:
            params = {
                "q": query,
                "count": max_per_page,
                "offset": offset,
                "search_lang": "en",
                "spellcheck": "0",
            }
            headers = {
                "X-Subscription-Token": self.api_key,
                "User-Agent": self.user_agent,
            }
            try:
                self.throttle.wait()
                response = self.session.get(
                    self.base_url,
                    headers=headers,
                    params=params,
                    timeout=30,
                )
                if response.status_code == 429:
                    self.throttle.backoff()
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException:
                break

            web_results = data.get("web", {}).get("results", [])
            added = 0
            for item in web_results:
                url = item.get("url")
                title = item.get("title", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append((url, title))
                added += 1
                if len(results) >= max_total:
                    break
            if added == 0:
                break
            offset += max_per_page
            self.throttle.success()
        return results


class ESGCrawler:
    def __init__(
        self,
        args: argparse.Namespace,
        manifest_path: str,
        raw_root: str,
        seen_urls: Set[str],
        seen_hashes: Set[str],
        company_map: Dict[str, str],
    ) -> None:
        self.args = args
        self.manifest_path = manifest_path
        self.raw_root = raw_root
        self.company_map = company_map  # 儲存公司名稱映射
        self.user_agent = args.user_agent or DEFAULT_USER_AGENT
        self.session = self._build_session()
        self.throttle = AdaptiveThrottle(args.throttle_sec, args.max_throttle_sec)
        self.robots_cache = RobotsCache(self.session, self.user_agent, self.throttle, args.respect_robots)
        self.search_client = BraveSearchClient(self.session, args.brave_key, self.throttle, self.user_agent)
        self.seen_urls = seen_urls
        self.seen_hashes = seen_hashes
        self.writer = self._open_manifest()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.args.retry_total,
            read=self.args.retry_total,
            connect=self.args.retry_total,
            backoff_factor=self.args.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": self.user_agent})
        return session

    def _open_manifest(self) -> csv.DictWriter:
        handle = open(self.manifest_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if handle.tell() == 0:
            writer.writeheader()
        self.manifest_handle = handle
        return writer

    def close(self) -> None:
        if hasattr(self, "manifest_handle") and self.manifest_handle:
            self.manifest_handle.flush()
            self.manifest_handle.close()

    def run(self, tasks: Iterable[QueryTask]) -> None:
        tasks_list = list(tasks)
        total_tasks = len(tasks_list)

        # 統計信息
        current_ticker = None
        ticker_index = 0
        total_tickers = len(set(t.ticker for t in tasks_list))
        downloaded_count = 0
        skipped_count = 0
        error_count = 0

        print(f"\n開始爬取 {total_tickers} 家公司的 ESG 報告...", flush=True)
        print(f"總任務數: {total_tasks}", flush=True)
        print("=" * 50, flush=True)

        for i, task in enumerate(tasks_list, 1):
            # 當開始處理新公司時顯示進度
            if task.ticker != current_ticker:
                if current_ticker is not None:
                    print(f"  完成 {current_ticker}", flush=True)
                current_ticker = task.ticker
                ticker_index += 1
                print(f"\n處理 {ticker_index}/{total_tickers}: {task.ticker}", flush=True)

            # 處理查詢並獲取統計
            result = self._process_query(task)
            if result == "downloaded":
                downloaded_count += 1
            elif result == "skipped":
                skipped_count += 1
            elif result == "error":
                error_count += 1

        print(f"\n{'=' * 50}", flush=True)
        print(f"爬取完成!", flush=True)
        print(f"成功下載: {downloaded_count} 個 PDF", flush=True)
        print(f"跳過: {skipped_count} 個", flush=True)
        print(f"錯誤: {error_count} 個", flush=True)
        print(f"{'=' * 50}", flush=True)

    def _process_query(self, task: QueryTask) -> str:
        """
        處理單個查詢任務

        返回:
            str: 'downloaded', 'skipped', 'error', 或 'no_results'
        """
        results = self.search_client.search(task.query, self.args.max_results, self.args.max_results_total)
        if not results:
            print(f"  查詢: {task.ticker} {task.year} ... 找到 0 個結果", flush=True)
            return "no_results"

        print(f"  查詢: {task.ticker} {task.year} ... 找到 {len(results)} 個結果", flush=True)

        downloaded_in_query = False
        skipped_in_query = False
        error_in_query = False
        for url, title in results:
            normalized_url = url.lower()

            # 網域檢查：驗證 URL 是否與公司相關
            company_name_for_check = self.company_map.get(task.ticker, task.ticker)
            if not is_domain_related(company_name_for_check, url):
                self._write_manifest(
                    task,
                    url=url,
                    final_url=url,
                    title=title,
                    status="skipped",
                    status_detail="domain_mismatch",
                    http_status="",
                    mime="",
                    content_bytes="",
                    digest="",
                    error_reason=f"Domain '{get_clean_domain(url)}' not related to '{company_name_for_check}'",
                    robots_allowed="",
                    content_disposition="",
                )
                skipped_in_query = True
                continue

            if normalized_url in self.seen_urls:
                self._write_manifest(
                    task,
                    url=url,
                    final_url=url,
                    title=title,
                    status="skipped",
                    status_detail="duplicate_url",
                    http_status="",
                    mime="",
                    content_bytes="",
                    digest="",
                    error_reason="",
                    robots_allowed="",
                    content_disposition="",
                )
                skipped_in_query = True
                continue

            allowed, robots_allowed = self.robots_cache.allowed(url)
            robots_flag = "true" if robots_allowed else ("false" if robots_allowed is False else "unknown")
            if not allowed and self.args.respect_robots:
                self._write_manifest(
                    task,
                    url=url,
                    final_url=url,
                    title=title,
                    status="skipped",
                    status_detail="robots_disallow",
                    http_status="",
                    mime="",
                    content_bytes="",
                    digest="",
                    error_reason="",
                    robots_allowed=robots_flag,
                    content_disposition="",
                )
                skipped_in_query = True
                continue

            fetch_result = self._download_with_html_probe(url, title)
            if fetch_result is None:
                error_in_query = True
                continue
            (
                final_url,
                content_bytes,
                headers,
                status_code,
                status,
                status_detail,
                error_reason,
            ) = fetch_result

            if content_bytes:
                digest = sha256_bytes(content_bytes)
                if digest in self.seen_hashes:
                    self._write_manifest(
                        task,
                        url=url,
                        final_url=final_url,
                        title=title,
                        status="skipped",
                        status_detail="duplicate_sha256",
                        http_status=status_code,
                        mime=headers.get("Content-Type", ""),
                        content_bytes=len(content_bytes),
                        digest=digest,
                        error_reason="",
                        robots_allowed=robots_flag,
                        content_disposition=headers.get("Content-Disposition", ""),
                    )
                    skipped_in_query = True
                    continue
                saved_path = self._save_pdf(task.ticker, task.year, digest, content_bytes)
                self.seen_hashes.add(digest)
                self.seen_urls.add(normalized_url)
                self.seen_urls.add(final_url.lower())
                self._write_manifest(
                    task,
                    url=url,
                    final_url=final_url,
                    title=title,
                    status=status,
                    status_detail=status_detail,
                    http_status=status_code,
                    mime=headers.get("Content-Type", ""),
                    content_bytes=len(content_bytes),
                    digest=digest,
                    error_reason=error_reason,
                    robots_allowed=robots_flag,
                    content_disposition=headers.get("Content-Disposition", ""),
                    extra={"saved_path": saved_path},
                )
                # 顯示下載成功信息
                file_size_mb = len(content_bytes) / (1024 * 1024)
                print(f"    ✓ 下載: {task.ticker}_{task.year}_{digest[:8]}.pdf ({file_size_mb:.2f} MB)", flush=True)
                downloaded_in_query = True
            else:
                self._write_manifest(
                    task,
                    url=url,
                    final_url=final_url,
                    title=title,
                    status=status,
                    status_detail=status_detail,
                    http_status=status_code,
                    mime=headers.get("Content-Type", ""),
                    content_bytes="",
                    digest="",
                    error_reason=error_reason,
                    robots_allowed=robots_flag,
                    content_disposition=headers.get("Content-Disposition", ""),
                )
                error_in_query = True

        # 返回此查詢的狀態
        if downloaded_in_query:
            return "downloaded"
        elif error_in_query:
            return "error"
        elif skipped_in_query:
            return "skipped"
        else:
            return "no_results"

    def _download_with_html_probe(self, url: str, title: str) -> Optional[Tuple[str, Optional[bytes], Dict[str, str], str, str, str, str]]:
        visited: Set[str] = set()
        current_url = url
        attempt = 0
        max_depth = self.args.max_html_probe_depth

        # 用於評分的關鍵字 (從 DEFAULT_REPORT_KEYWORDS 和 title 獲取)
        probe_keywords = list(DEFAULT_REPORT_KEYWORDS) + title.split()
        # 從 title 中提取年份
        year_match = re.search(r"(201\d|202\d)", title)
        if year_match:
            probe_keywords.append(year_match.group(1))

        while attempt <= max_depth:
            try:
                self.throttle.wait()
                response = self.session.get(
                    current_url,
                    timeout=self.args.http_timeout,
                    stream=True,
                    allow_redirects=True,
                )
                if response.status_code == 429:
                    self.throttle.backoff()
                else:
                    self.throttle.success()
            except requests.exceptions.RequestException as exc:
                return (
                    current_url,
                    None,
                    {},
                    "",
                    "error",
                    "request_exception",
                    str(exc),
                )

            headers = dict(response.headers)
            status_code = str(response.status_code)
            final_url = response.url

            if response.status_code >= 400:
                response.close()
                return (
                    final_url,
                    None,
                    headers,
                    status_code,
                    "error",
                    "http_error",
                    f"{response.status_code}",
                )

            content_disposition = headers.get("Content-Disposition", "")
            mime_is_pdf = is_pdf_mime(response.headers)
            url_looks_pdf = is_pdf_like_url(final_url)
            if mime_is_pdf or url_looks_pdf or ".pdf" in content_disposition.lower():
                payload, reason = self._collect_payload(response)
                if payload is None:
                    response.close()
                    return (
                        final_url,
                        None,
                        headers,
                        status_code,
                        "skipped",
                        reason,
                        "",
                    )
                if not payload.startswith(PDF_SIGNATURE):
                    response.close()
                    return (
                        final_url,
                        None,
                        headers,
                        status_code,
                        "skipped",
                        "skipped_non_pdf_payload",
                        "",
                    )
                return (
                    final_url,
                    payload,
                    headers,
                    status_code,
                    "downloaded",
                    "ok",
                    "",
                )

            try:
                html_snippet = response.content[:512000]
            except Exception:
                html_snippet = b""
            finally:
                response.close()
            if not html_snippet:
                return (
                    final_url,
                    None,
                    headers,
                    status_code,
                    "skipped",
                    "empty_payload",
                    "",
                )
            try:
                html_text = html_snippet.decode(response.encoding or "utf-8", errors="ignore")
            except Exception:
                html_text = html_snippet.decode("utf-8", errors="ignore")

            # 使用新的評分機制
            parser = PDFLinkExtractor()
            try:
                parser.feed(html_text)
            except Exception:
                pass  # 解析 HTML 失敗

            scored_links = parser.get_scored_pdf_links(final_url, probe_keywords)

            if not scored_links:
                # 找不到任何有價值的 PDF 連結
                return (
                    final_url,
                    None,
                    headers,
                    status_code,
                    "skipped",
                    "html_without_pdf",
                    "",
                )

            # 選擇分數最高的連結
            pdf_link = scored_links[0][1]  # (score, url)

            if pdf_link in visited:
                return (
                    final_url,
                    None,
                    headers,
                    status_code,
                    "skipped",
                    "html_pdf_loop",
                    "",
                )
            visited.add(pdf_link)
            current_url = pdf_link
            attempt += 1

        return (
            current_url,
            None,
            {},
            "",
            "skipped",
            "max_probe_depth",
            "",
        )

    def _collect_payload(self, response: requests.Response) -> Tuple[Optional[bytes], str]:
        max_bytes = self.args.max_pdf_bytes
        chunks: List[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                response.close()
                return None, "skipped_too_large"
            chunks.append(chunk)
        response.close()
        payload = b"".join(chunks)
        if not payload:
            return None, "empty_payload"
        return payload, "ok"

    def _save_pdf(self, ticker: str, year: int, digest: str, payload: bytes) -> str:
        out_dir = os.path.join(self.raw_root, ticker, str(year))
        os.makedirs(out_dir, exist_ok=True)
        filename = sanitize_filename(f"{ticker}_{year}_{digest[:8]}.pdf")
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "wb") as handle:
            handle.write(payload)
        return out_path

    def _write_manifest(
        self,
        task: QueryTask,
        *,
        url: str,
        final_url: str,
        title: str,
        status: str,
        status_detail: str,
        http_status: str,
        mime: str,
        content_bytes,
        digest: str,
        error_reason: str,
        robots_allowed: str,
        content_disposition: str,
        extra: Optional[Dict[str, str]] = None,
    ) -> None:
        row = OrderedDict(
            [
                ("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                ("ticker", task.ticker),
                ("year", str(task.year)),
                ("query", task.query),
                ("url", url),
                ("final_url", final_url),
                ("title", title),
                ("source", "brave"),
                ("mime", mime),
                ("bytes", content_bytes if isinstance(content_bytes, int) else ""),
                ("sha256", digest),
                ("status", status),
                ("status_detail", status_detail),
                ("http_status", http_status),
                ("error_reason", error_reason),
                ("robots_allowed", robots_allowed),
                ("content_disposition", content_disposition),
                ("saved_path", ""),
            ]
        )
        if extra:
            for key, value in extra.items():
                if key in MANIFEST_COLUMNS:
                    row[key] = value
        self.writer.writerow(row)
        self.manifest_handle.flush()


def make_queries(
    ticker: str,
    company_name: Optional[str],
    years: Sequence[int],
    report_keywords: Sequence[str],
) -> List[QueryTask]:
    name = company_name or ticker
    quoted_keywords = [f'"{kw}"' for kw in report_keywords if kw]
    if not quoted_keywords:
        quoted_keywords = [f'"{kw}"' for kw in DEFAULT_REPORT_KEYWORDS]
    base_keywords = "(" + " OR ".join(quoted_keywords) + ")"
    blocklist = " ".join(f"-site:{domain}" for domain in AGGREGATOR_BLOCKLIST)

    # 較寬鬆的學術排除詞 (只排除最明顯的學術網站，不在標題中排除)
    academic_blocklist = "-site:researchgate.net -site:scholar.google.com -site:ssrn.com"

    queries: List[QueryTask] = []
    seen_queries: Set[str] = set()
    for year in years:
        for delta in (0, -1, 1):
            target_year = year + delta
            if target_year < 1900:
                continue

            # 使用 OR 邏輯結合公司名稱和 ticker，使查詢更寬鬆
            # 只有關鍵詞使用引號，公司名稱不加引號以避免過於嚴格
            company_identifier = f'({name} OR {ticker})'

            # 主查詢：結合公司識別符和關鍵詞
            main_query = f'{company_identifier} {base_keywords} {target_year} filetype:pdf {blocklist} {academic_blocklist}'

            # 備用查詢：如果主查詢太寬鬆，這個更精確 (使用引號)
            # 但是使用 OR 而非兩個獨立查詢，避免重複搜索
            fallback_query = f'("{name}" OR "{ticker}") {base_keywords} "{target_year}" filetype:pdf {blocklist} {academic_blocklist}'

            q_variants = [main_query, fallback_query]

            for q in q_variants:
                normalized = q.strip()
                if normalized in seen_queries:
                    continue
                seen_queries.add(normalized)
                queries.append(QueryTask(ticker=ticker, year=year, query=normalized))
    return queries


def build_query_tasks(
    tickers: Sequence[str],
    company_map: Dict[str, str],
    years: Sequence[int],
    report_keywords: Sequence[str],
) -> List[QueryTask]:
    tasks: List[QueryTask] = []
    for ticker in tickers:
        company_name = company_map.get(ticker)
        tasks.extend(make_queries(ticker, company_name, years, report_keywords))
    return tasks


def create_args_from_config() -> argparse.Namespace:
    """從配置文件建立參數物件（用於向後相容）"""
    from .config_loader import get_config
    import os

    config = get_config()

    # 建立一個 Namespace 物件來模擬 argparse 的行為
    args = argparse.Namespace()

    # 從配置文件讀取參數
    args.root = config.root
    args.start_year = config.crawler_start_year
    args.end_year = config.crawler_end_year
    args.brave_key = os.getenv('BRAVE_API_KEY')  # 從環境變數讀取
    args.company_map = config.crawler_company_map
    args.max_results = config.crawler_max_results
    args.max_results_total = config.crawler_max_results_total
    args.throttle_sec = config.crawler_throttle_sec
    args.max_throttle_sec = config.crawler_max_throttle_sec
    args.retry_total = config.crawler_retry_total
    args.retry_backoff = config.crawler_retry_backoff
    args.http_timeout = config.crawler_http_timeout
    args.max_pdf_bytes = config.crawler_max_pdf_bytes
    args.user_agent = config.crawler_user_agent
    args.respect_robots = config.crawler_respect_robots
    args.max_html_probe_depth = config.crawler_max_html_probe_depth

    return args


def main() -> None:
    """主函數 - 使用配置文件替代 CLI 參數"""
    print("注意: 本爬蟲現在使用 config/config.yaml 進行配置", flush=True)
    print("API Key 請設定在 .env 文件的 BRAVE_API_KEY 環境變數中\n", flush=True)

    # 從配置文件建立參數
    args = create_args_from_config()

    # 檢查必要的 API Key
    if not args.brave_key:
        print("錯誤: 未找到 BRAVE_API_KEY 環境變數", flush=True)
        print("請在 .env 文件中設定: BRAVE_API_KEY=your_api_key", flush=True)
        return

    # 驗證參數
    args.max_results = max(1, min(args.max_results, 20))
    args.max_results_total = max(args.max_results, args.max_results_total)
    if args.max_pdf_bytes <= 0:
        args.max_pdf_bytes = 40 * 1024 * 1024

    # 從配置讀取路徑
    from .config_loader import get_config
    config = get_config()

    root = args.root
    sp500_csv = config.sp500_csv
    manifest_path = config.manifest_path
    raw_root = config.raw_data_path

    ensure_manifest_header(manifest_path)
    seen_urls, seen_hashes = load_manifest_index(manifest_path)
    years = list(range(args.start_year, args.end_year + 1))
    keywords_config = load_keywords(root)
    report_keywords = keywords_config.get("report_keywords", DEFAULT_REPORT_KEYWORDS)
    if not isinstance(report_keywords, (list, tuple)):
        report_keywords = DEFAULT_REPORT_KEYWORDS
    tickers = read_sp500_csv(sp500_csv)
    company_map = read_company_map(args.company_map)
    tasks = build_query_tasks(tickers, company_map, years, report_keywords)

    crawler = ESGCrawler(args, manifest_path, raw_root, seen_urls, seen_hashes, company_map)
    try:
        crawler.run(tasks)
    finally:
        crawler.close()
    print("Done. Manifest at:", manifest_path, flush=True)


if __name__ == "__main__":
    main()

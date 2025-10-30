#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
主執行腳本
"""
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.data_loader import DataLoader, load_corpus
from src.topic_modeler import TopicModeler, load_topic_model
from src.esg_crawler import (
    read_sp500_csv, read_company_map, load_keywords,
    make_queries, brave_search, is_pdf_url, download_pdf,
    safe_filename, sha256_bytes, content_type_pdf
)

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('esg_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def stage0_setup():
    """階段0: 環境設定與專案結構"""
    logger.info("=" * 50)
    logger.info("階段 0: 環境設定與專案結構")
    logger.info("=" * 50)

    # 檢查目錄結構
    required_dirs = [
        'data/raw',
        'data/metadata',
        'data/processed_corpus',
        'data/models',
        'data/results',
        'src',
        'notebooks',
        'tests'
    ]

    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"檢查目錄: {dir_path} ✓")

    # 檢查環境變數
    required_env_vars = ['OPENAI_API_KEY']

    for var in required_env_vars:
        if not os.getenv(var):
            logger.warning(f"環境變數 {var} 未設定!")
        else:
            logger.info(f"環境變數 {var} 已設定 ✓")

    logger.info("階段 0 完成\n")


def stage0_crawl_pdfs(start_year=2017, end_year=2018, max_results=8, throttle_sec=1.0):
    """階段0可選: 使用 Brave Search API 爬取 ESG PDF 報告"""
    logger.info("=" * 50)
    logger.info("階段 0 (可選): 爬取 ESG PDF 報告")
    logger.info("=" * 50)

    # 檢查 Brave API Key
    brave_key = os.getenv('BRAVE_API_KEY')
    if not brave_key:
        logger.error("環境變數 BRAVE_API_KEY 未設定！")
        logger.error("請在 .env 文件中添加: BRAVE_API_KEY=your_api_key")
        return

    import csv
    import time

    # 設置路徑
    root = "."
    sp500_csv = os.path.join(root, "data", "sp500_2017-01-27.csv")
    manifest_path = os.path.join(root, "data", "metadata", "esg_manifest.csv")
    raw_root = os.path.join(root, "data", "raw")

    # 檢查 SP500 CSV 是否存在
    if not os.path.exists(sp500_csv):
        logger.error(f"找不到 SP500 CSV 文件: {sp500_csv}")
        logger.error("請確保該文件存在後再運行爬蟲")
        return

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    os.makedirs(raw_root, exist_ok=True)

    years = list(range(start_year, end_year + 1))
    kw = load_keywords(root)
    report_keywords = kw.get("report_keywords", [])
    tickers = read_sp500_csv(sp500_csv)
    company_map = read_company_map(None)  # 可選的公司名稱映射

    logger.info(f"爬取年份: {start_year}-{end_year}")
    logger.info(f"公司數量: {len(tickers)}")
    logger.info(f"報告關鍵詞: {report_keywords}")

    write_header = not os.path.exists(manifest_path)
    mf = open(manifest_path, "a", newline="", encoding="utf-8")
    w = csv.writer(mf)
    if write_header:
        w.writerow(["ticker", "year", "url", "title", "source", "mime", "bytes", "sha256", "status"])

    total_downloaded = 0
    total_skipped = 0
    total_errors = 0

    for idx, tkr in enumerate(tickers, 1):
        logger.info(f"處理 {idx}/{len(tickers)}: {tkr}")
        name = company_map.get(tkr, None)
        queries = make_queries(tkr, name, years, report_keywords)

        for q in queries:
            try:
                results = brave_search(q, brave_key, max_results=max_results)
                logger.info(f"  查詢: {q[:80]}... 找到 {len(results)} 個結果")
            except Exception as e:
                logger.error(f"  Brave Search 錯誤: {e}")
                time.sleep(throttle_sec * 2)
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
                        total_downloaded += 1
                        logger.info(f"    ✓ 下載: {fname} ({len(blob)} bytes)")
                    else:
                        w.writerow([tkr, year_in_q, link, title, "brave", (headers or {}).get("Content-Type", ""), "", "", "skipped_non_pdf"])
                        total_skipped += 1
                except Exception as e:
                    w.writerow([tkr, year_in_q, link, title, "brave", "", "", "", f"error:{type(e).__name__}"])
                    total_errors += 1
                    logger.error(f"    ✗ 下載錯誤: {e}")

                mf.flush()
                time.sleep(throttle_sec)

        time.sleep(throttle_sec * 2)

    mf.close()

    logger.info("=" * 50)
    logger.info(f"爬取完成!")
    logger.info(f"成功下載: {total_downloaded} 個 PDF")
    logger.info(f"跳過: {total_skipped} 個")
    logger.info(f"錯誤: {total_errors} 個")
    logger.info(f"Manifest 位置: {manifest_path}")
    logger.info("=" * 50)


def stage1_data_loading(use_semantic_chunking=False):
    """階段1: 資料載入與預處理"""
    logger.info("=" * 50)
    logger.info("階段 1: 資料載入與預處理")
    logger.info("=" * 50)

    # 初始化DataLoader
    data_loader = DataLoader(
        raw_data_path="./data/raw",
        processed_data_path="./data/processed_corpus",
        min_sentence_length=50,
        max_chunk_tokens=512
    )

    # 探索資料集
    stats = data_loader.explore_dataset()
    logger.info(f"資料集統計: {stats}")

    # 建立語料庫
    corpus_df = data_loader.build_corpus(
        output_filename="corpus.csv",
        use_semantic_chunking=use_semantic_chunking
    )

    logger.info(f"語料庫建立完成: {len(corpus_df)} 個文本塊")
    logger.info("階段 1 完成\n")

    return corpus_df


def stage2_topic_modeling():
    """階段2: 主題建模"""
    logger.info("=" * 50)
    logger.info("階段 2: 主題建模")
    logger.info("=" * 50)

    # 載入語料庫
    corpus_df = load_corpus("./data/processed_corpus/corpus.csv")
    texts = corpus_df['text'].tolist()

    logger.info(f"載入 {len(texts)} 個文本")

    # 初始化TopicModeler
    topic_modeler = TopicModeler(
        models_path="./data/models",
        results_path="./data/results",
        embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        embedding_batch_size=100,
        n_neighbors=15,
        n_components=5,
        min_cluster_size=40,
        min_samples=10,
        verbose=True,
        n_jobs=20,  # 使用20個CPU核心
        low_memory=False  # 高性能模式
    )

    # 生成嵌入
    embeddings = topic_modeler.generate_embeddings(
        texts,
        cache_name="embeddings"
    )

    logger.info(f"嵌入向量形狀: {embeddings.shape}")

    # 訓練模型
    topic_model, topic_info, corpus_with_topics, probs = topic_modeler.train_initial_model(
        texts,
        embeddings,
        model_name="phase2_bertopic_model"
    )

    logger.info(f"發現 {len(topic_info)} 個主題")

    # 導出JSON摘要
    json_path = topic_modeler.export_topic_summary_json(
        topic_model,
        output_filename="phase2_topic_summary.json"
    )

    logger.info(f"主題摘要已導出至: {json_path}")
    logger.info("階段 2 完成\n")

    return topic_model, json_path


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='ESG報告主題建模與數位韌性量化框架')

    parser.add_argument(
        '--stage',
        type=str,
        choices=['0', '1', '2', 'all'],
        default='all',
        help='執行階段: 0=環境設定, 1=資料載入, 2=主題建模, all=全部'
    )

    parser.add_argument(
        '--semantic-chunking',
        action='store_true',
        help='使用語義分塊（需要OpenAI API）'
    )

    # Brave 爬蟲相關參數
    parser.add_argument(
        '--crawl',
        action='store_true',
        help='執行 Brave Search API 爬蟲（階段 0 可選）'
    )

    parser.add_argument(
        '--start-year',
        type=int,
        default=2017,
        help='爬蟲起始年份（預設: 2017）'
    )

    parser.add_argument(
        '--end-year',
        type=int,
        default=2018,
        help='爬蟲結束年份（預設: 2018）'
    )

    parser.add_argument(
        '--max-results',
        type=int,
        default=8,
        help='每次搜尋的最大結果數（預設: 8）'
    )

    parser.add_argument(
        '--throttle-sec',
        type=float,
        default=1.0,
        help='API 請求間隔秒數（預設: 1.0）'
    )

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("ESG報告主題建模與數位韌性量化框架")
    logger.info("=" * 50)

    try:
        if args.stage in ['0', 'all']:
            stage0_setup()

            # 如果指定了 --crawl 參數，則執行爬蟲
            if args.crawl:
                stage0_crawl_pdfs(
                    start_year=args.start_year,
                    end_year=args.end_year,
                    max_results=args.max_results,
                    throttle_sec=args.throttle_sec
                )

        if args.stage in ['1', 'all']:
            stage1_data_loading(use_semantic_chunking=args.semantic_chunking)

        if args.stage in ['2', 'all']:
            stage2_topic_modeling()

        logger.info("=" * 50)
        logger.info("所有階段執行完成!")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"執行時出錯: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

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

from src.config_loader import get_config, load_keywords
from src.data_loader import DataLoader, load_corpus
from src.topic_modeler import TopicModeler, load_topic_model
from src.esg_crawler import (
    read_sp500_csv, read_company_map,
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

# 載入配置
config = get_config()


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


def stage0_crawl_pdfs():
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

    # 從配置讀取參數
    root = config.root
    sp500_csv = config.sp500_csv
    manifest_path = config.manifest_path
    raw_root = config.raw_data_path
    start_year = config.crawler_start_year
    end_year = config.crawler_end_year
    max_results = config.crawler_max_results
    throttle_sec = config.crawler_throttle_sec

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
    company_map = read_company_map(config.crawler_company_map)

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


def stage1_data_loading():
    """階段1: 資料載入與預處理"""
    logger.info("=" * 50)
    logger.info("階段 1: 資料載入與預處理")
    logger.info("=" * 50)

    # 從配置讀取參數並初始化DataLoader
    data_loader = DataLoader(
        raw_data_path=config.raw_data_path,
        processed_data_path=config.processed_data_path,
        min_sentence_length=config.data_loader_min_sentence_length,
        max_chunk_tokens=config.data_loader_max_chunk_tokens,
        boilerplate_keywords=config.data_loader_boilerplate_keywords
    )

    # 探索資料集
    stats = data_loader.explore_dataset()
    logger.info(f"資料集統計: {stats}")

    # 建立語料庫（使用配置中的 semantic_chunking 設定）
    corpus_df = data_loader.build_corpus(
        output_filename="corpus.csv",
        use_semantic_chunking=config.semantic_chunking
    )

    logger.info(f"語料庫建立完成: {len(corpus_df)} 個文本塊")
    logger.info("階段 1 完成\n")

    return corpus_df


def stage2_topic_modeling():
    """階段2: 主題建模"""
    logger.info("=" * 50)
    logger.info("階段 2: 主題建模")
    logger.info("=" * 50)

    # 從配置讀取路徑並載入語料庫
    corpus_path = os.path.join(config.processed_data_path, "corpus.csv")
    corpus_df = load_corpus(corpus_path)
    texts = corpus_df['text'].tolist()

    logger.info(f"載入 {len(texts)} 個文本")

    # 從配置讀取參數並初始化TopicModeler
    topic_modeler = TopicModeler(
        models_path=config.models_path,
        results_path=config.results_path,
        embedding_model=config.topic_modeler_embedding_model,
        embedding_batch_size=config.topic_modeler_embedding_batch_size,
        n_neighbors=config.topic_modeler_umap_n_neighbors,
        n_components=config.topic_modeler_umap_n_components,
        min_dist=config.topic_modeler_umap_min_dist,
        metric=config.topic_modeler_umap_metric,
        min_cluster_size=config.topic_modeler_hdbscan_min_cluster_size,
        min_samples=config.topic_modeler_hdbscan_min_samples,
        cluster_selection_method=config.topic_modeler_hdbscan_cluster_selection_method,
        n_jobs=config.topic_modeler_computing_n_jobs,
        low_memory=config.topic_modeler_computing_low_memory,
        verbose=config.topic_modeler_computing_verbose
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
    parser = argparse.ArgumentParser(
        description='ESG報告主題建模與數位韌性量化框架',
        epilog='注意: 除了 --stage 參數外，其他所有參數請在 config/config.yaml 中設定'
    )

    # 只保留 stage 參數
    parser.add_argument(
        '--stage',
        type=str,
        choices=['0', '1', '2', 'all'],
        default='all',
        help='執行階段: 0=環境設定, 1=資料載入, 2=主題建模, all=全部'
    )

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("ESG報告主題建模與數位韌性量化框架")
    logger.info("=" * 50)
    logger.info(f"配置文件: {config.config_path}")
    logger.info(f"執行階段: {args.stage}")

    try:
        if args.stage in ['0', 'all']:
            stage0_setup()

            # 根據配置決定是否執行爬蟲
            if config.crawl:
                stage0_crawl_pdfs()

        if args.stage in ['1', 'all']:
            stage1_data_loading()

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

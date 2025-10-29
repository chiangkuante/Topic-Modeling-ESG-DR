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
        Verbose=True,
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

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("ESG報告主題建模與數位韌性量化框架")
    logger.info("=" * 50)

    try:
        if args.stage in ['0', 'all']:
            stage0_setup()

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

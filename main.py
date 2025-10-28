# 主執行腳本
# ESG 報告主題建模與數位韌性量化框架

import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

from src.utils import setup_logging, load_config
from src.crawler import Crawler
from src.topic_modeler import TopicModeler
from src.mapper import TopicMapper
from src.scorer import DigitalResilienceScorer

logger = logging.getLogger(__name__)


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='ESG 報告主題建模與數位韌性量化框架'
    )
    parser.add_argument(
        '--run-crawler',
        action='store_true',
        help='執行爬蟲模組'
    )
    parser.add_argument(
        '--run-modeling',
        action='store_true',
        help='執行主題建模模組'
    )
    parser.add_argument(
        '--run-mapping',
        action='store_true',
        help='執行主題映射模組'
    )
    parser.add_argument(
        '--run-scoring',
        action='store_true',
        help='執行評分模組'
    )
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='執行完整流程'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='配置檔案路徑'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日誌級別'
    )
    return parser.parse_args()


def main():
    """主執行函數"""
    args = parse_args()

    # 設定日誌
    setup_logging(args.log_level)
    logger.info("開始執行 ESG 報告主題建模與數位韌性量化框架")

    # 載入環境變數
    load_dotenv()
    logger.info("環境變數載入完成")

    # 載入配置
    # config_path = Path(args.config)
    # config = load_config(config_path) if config_path.exists() else {}

    # 暫時使用預設配置
    config = {
        'data_dir': Path('data'),
        'raw_reports_dir': Path('data/raw_reports'),
        'processed_corpus_dir': Path('data/processed_corpus'),
        'models_dir': Path('data/models'),
        'results_dir': Path('data/results')
    }

    # 執行爬蟲
    if args.run_crawler or args.run_all:
        logger.info("=" * 50)
        logger.info("階段 1: 爬蟲與預處理")
        logger.info("=" * 50)
        crawler = Crawler(config)
        # crawler.download_reports(tickers, start_date, end_date)
        # crawler.process_files()
        # corpus_path = crawler.build_corpus()
        logger.info("爬蟲階段完成 (尚未實現)")

    # 執行主題建模
    if args.run_modeling or args.run_all:
        logger.info("=" * 50)
        logger.info("階段 2: 主題建模")
        logger.info("=" * 50)
        modeler = TopicModeler(config)
        # df = modeler.load_corpus(corpus_path)
        # embeddings = modeler.generate_embeddings(df['text'].tolist())
        # modeler.train_model(df['text'].tolist(), embeddings)
        logger.info("主題建模階段完成 (尚未實現)")

    # 執行主題映射
    if args.run_mapping or args.run_all:
        logger.info("=" * 50)
        logger.info("階段 3: 主題映射")
        logger.info("=" * 50)
        mapper = TopicMapper(config)
        # mapper.load_model(model_path)
        # topic_map = mapper.map_topics_to_dimensions(dimensions)
        logger.info("主題映射階段完成 (尚未實現)")

    # 執行評分
    if args.run_scoring or args.run_all:
        logger.info("=" * 50)
        logger.info("階段 4: 評分與 DRI 計算")
        logger.info("=" * 50)
        scorer = DigitalResilienceScorer(config)
        # topic_scores = scorer.score_topics(topic_map)
        # df_scores = scorer.calculate_document_scores(df, topic_scores)
        # df_dri = scorer.calculate_dri(df_scores, dimension_weights)
        logger.info("評分階段完成 (尚未實現)")

    logger.info("=" * 50)
    logger.info("所有任務執行完成")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_loader.py
配置載入模組 - 從 config.yaml 載入所有配置參數
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """配置管理類別"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        參數:
            config_path: 配置文件路徑，如果為 None 則使用預設路徑
        """
        if config_path is None:
            # 預設配置路徑
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        logger.info(f"配置已載入: {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """載入配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"配置文件格式錯誤: {self.config_path}")

        return config

    def get(self, *keys, default=None) -> Any:
        """
        獲取配置值（支援多層級鍵）

        參數:
            *keys: 配置鍵的路徑，例如 get('crawler', 'start_year')
            default: 預設值

        返回:
            配置值
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    # ===== 路徑相關配置 =====
    @property
    def root(self) -> str:
        return self.get('paths', 'root', default='.')

    @property
    def raw_data_path(self) -> str:
        return self.get('paths', 'raw_data', default='./data/raw')

    @property
    def processed_data_path(self) -> str:
        return self.get('paths', 'processed_data', default='./data/processed_corpus')

    @property
    def models_path(self) -> str:
        return self.get('paths', 'models', default='./data/models')

    @property
    def results_path(self) -> str:
        return self.get('paths', 'results', default='./data/results')

    @property
    def sp500_csv(self) -> str:
        return self.get('paths', 'sp500_csv', default='./data/sp500_2017-01-27.csv')

    @property
    def manifest_path(self) -> str:
        return self.get('paths', 'manifest', default='./data/metadata/esg_manifest.csv')

    # ===== Pipeline 配置 =====
    @property
    def semantic_chunking(self) -> bool:
        return self.get('pipeline', 'semantic_chunking', default=False)

    @property
    def crawl(self) -> bool:
        return self.get('pipeline', 'crawl', default=False)

    # ===== Crawler 配置 =====
    @property
    def crawler_start_year(self) -> int:
        return self.get('crawler', 'start_year', default=2017)

    @property
    def crawler_end_year(self) -> int:
        return self.get('crawler', 'end_year', default=2018)

    @property
    def crawler_company_map(self) -> Optional[str]:
        return self.get('crawler', 'company_map')

    @property
    def crawler_max_results(self) -> int:
        return self.get('crawler', 'max_results', default=20)

    @property
    def crawler_max_results_total(self) -> int:
        return self.get('crawler', 'max_results_total', default=60)

    @property
    def crawler_throttle_sec(self) -> float:
        return self.get('crawler', 'throttle_sec', default=1.0)

    @property
    def crawler_max_throttle_sec(self) -> float:
        return self.get('crawler', 'max_throttle_sec', default=8.0)

    @property
    def crawler_retry_total(self) -> int:
        return self.get('crawler', 'retry_total', default=5)

    @property
    def crawler_retry_backoff(self) -> float:
        return self.get('crawler', 'retry_backoff', default=1.0)

    @property
    def crawler_http_timeout(self) -> float:
        return self.get('crawler', 'http_timeout', default=45.0)

    @property
    def crawler_max_pdf_bytes(self) -> int:
        return self.get('crawler', 'max_pdf_bytes', default=40 * 1024 * 1024)

    @property
    def crawler_user_agent(self) -> str:
        return self.get('crawler', 'user_agent',
                       default='Mozilla/5.0 (compatible; ESGCrawler/1.1; +https://example.com/contact)')

    @property
    def crawler_respect_robots(self) -> bool:
        return self.get('crawler', 'respect_robots', default=False)

    @property
    def crawler_max_html_probe_depth(self) -> int:
        return self.get('crawler', 'max_html_probe_depth', default=2)

    # ===== Data Loader 配置 =====
    @property
    def data_loader_min_sentence_length(self) -> int:
        return self.get('data_loader', 'min_sentence_length', default=50)

    @property
    def data_loader_max_chunk_tokens(self) -> int:
        return self.get('data_loader', 'max_chunk_tokens', default=512)

    @property
    def data_loader_boilerplate_keywords(self) -> list:
        return self.get('data_loader', 'boilerplate_keywords', default=[
            "forward-looking statement",
            "safe harbor",
            "securities and exchange commission",
            "this report contains",
            "table of contents"
        ])

    # ===== Topic Modeler 配置 =====
    @property
    def topic_modeler_embedding_model(self) -> str:
        # 環境變數可以覆蓋配置文件
        return os.getenv('EMBEDDING_MODEL') or self.get('topic_modeler', 'embedding_model',
                                                         default='text-embedding-3-small')

    @property
    def topic_modeler_embedding_batch_size(self) -> int:
        return self.get('topic_modeler', 'embedding_batch_size', default=100)

    @property
    def topic_modeler_umap_n_neighbors(self) -> int:
        return self.get('topic_modeler', 'umap', 'n_neighbors', default=15)

    @property
    def topic_modeler_umap_n_components(self) -> int:
        return self.get('topic_modeler', 'umap', 'n_components', default=5)

    @property
    def topic_modeler_umap_min_dist(self) -> float:
        return self.get('topic_modeler', 'umap', 'min_dist', default=0.0)

    @property
    def topic_modeler_umap_metric(self) -> str:
        return self.get('topic_modeler', 'umap', 'metric', default='cosine')

    @property
    def topic_modeler_hdbscan_min_cluster_size(self) -> int:
        return self.get('topic_modeler', 'hdbscan', 'min_cluster_size', default=40)

    @property
    def topic_modeler_hdbscan_min_samples(self) -> int:
        return self.get('topic_modeler', 'hdbscan', 'min_samples', default=10)

    @property
    def topic_modeler_hdbscan_cluster_selection_method(self) -> str:
        return self.get('topic_modeler', 'hdbscan', 'cluster_selection_method', default='eom')

    @property
    def topic_modeler_vectorizer_max_features(self) -> int:
        return self.get('topic_modeler', 'vectorizer', 'max_features', default=1000)

    @property
    def topic_modeler_vectorizer_ngram_range(self) -> tuple:
        ngram = self.get('topic_modeler', 'vectorizer', 'ngram_range', default=[1, 2])
        return tuple(ngram)

    @property
    def topic_modeler_vectorizer_custom_stop_words(self) -> list:
        return self.get('topic_modeler', 'vectorizer', 'custom_stop_words', default=[])

    @property
    def topic_modeler_bertopic_top_n_words(self) -> int:
        return self.get('topic_modeler', 'bertopic', 'top_n_words', default=10)

    @property
    def topic_modeler_bertopic_verbose(self) -> bool:
        return self.get('topic_modeler', 'bertopic', 'verbose', default=True)

    @property
    def topic_modeler_bertopic_calculate_probabilities(self) -> bool:
        return self.get('topic_modeler', 'bertopic', 'calculate_probabilities', default=True)

    @property
    def topic_modeler_computing_n_jobs(self) -> int:
        return self.get('topic_modeler', 'computing', 'n_jobs', default=-1)

    @property
    def topic_modeler_computing_low_memory(self) -> bool:
        return self.get('topic_modeler', 'computing', 'low_memory', default=False)

    @property
    def topic_modeler_computing_verbose(self) -> bool:
        return self.get('topic_modeler', 'computing', 'verbose', default=False)


def load_keywords(root: str = ".") -> Dict[str, list]:
    """
    載入關鍵詞配置

    參數:
        root: 專案根目錄

    返回:
        關鍵詞字典
    """
    keywords_path = Path(root) / "config" / "keywords.yaml"

    if not keywords_path.exists():
        logger.warning(f"關鍵詞文件不存在: {keywords_path}，使用預設值")
        return {
            "report_keywords": [
                "sustainability report",
                "ESG report",
                "corporate responsibility report",
                "CSR report",
                "non-financial report",
                "sustainability accounting",
            ]
        }

    try:
        with open(keywords_path, 'r', encoding='utf-8') as f:
            keywords = yaml.safe_load(f)

        if isinstance(keywords, dict):
            logger.info(f"關鍵詞已載入: {keywords_path}")
            return keywords
        else:
            logger.warning(f"關鍵詞文件格式錯誤: {keywords_path}，使用預設值")
            return {"report_keywords": []}

    except Exception as e:
        logger.error(f"載入關鍵詞文件時出錯: {e}，使用預設值")
        return {"report_keywords": []}


# 全域配置實例（單例模式）
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    獲取全域配置實例

    參數:
        config_path: 配置文件路徑，如果為 None 則使用預設路徑

    返回:
        配置實例
    """
    global _global_config

    if _global_config is None:
        _global_config = Config(config_path)

    return _global_config

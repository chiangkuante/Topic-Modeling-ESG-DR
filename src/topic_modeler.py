#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topic_modeler.py
主題建模模組
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import time

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from openai import OpenAI
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 延遲載入配置，避免循環導入
_config = None

def _get_config():
    """延遲載入配置"""
    global _config
    if _config is None:
        try:
            from .config_loader import get_config
            _config = get_config()
        except Exception:
            _config = None
    return _config


class TopicModeler:
    """主題建模類別"""

    def __init__(
        self,
        models_path: Optional[str] = None,
        results_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_batch_size: Optional[int] = None,
        openai_api_key: Optional[str] = None,
        # UMAP參數
        n_neighbors: Optional[int] = None,
        n_components: Optional[int] = None,
        min_dist: Optional[float] = None,
        metric: Optional[str] = None,
        # HDBSCAN參數
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        cluster_selection_method: Optional[str] = None,
        # 並行計算參數
        n_jobs: Optional[int] = None,
        low_memory: Optional[bool] = None,
        verbose: Optional[bool] = None
    ):
        """
        初始化主題建模器

        參數:
            models_path: 模型保存路徑（如果為 None，從配置讀取）
            results_path: 結果保存路徑（如果為 None，從配置讀取）
            embedding_model: 嵌入模型名稱（如果為 None，從配置讀取）
            embedding_batch_size: 嵌入批次大小（如果為 None，從配置讀取）
            openai_api_key: OpenAI API金鑰
            n_neighbors: UMAP鄰居數（如果為 None，從配置讀取）
            n_components: UMAP維度數（如果為 None，從配置讀取）
            min_dist: UMAP最小距離（如果為 None，從配置讀取）
            metric: UMAP距離度量（如果為 None，從配置讀取）
            min_cluster_size: HDBSCAN最小聚類大小（如果為 None，從配置讀取）
            min_samples: HDBSCAN最小樣本數（如果為 None，從配置讀取）
            cluster_selection_method: HDBSCAN聚類選擇方法（如果為 None，從配置讀取）
            n_jobs: CPU核心數（如果為 None，從配置讀取）
            low_memory: 記憶體模式（如果為 None，從配置讀取）
            verbose: 詳細輸出（如果為 None，從配置讀取）
        """
        # 嘗試從配置讀取預設值
        config = _get_config()

        if models_path is None:
            models_path = config.models_path if config else "./data/models"
        if results_path is None:
            results_path = config.results_path if config else "./data/results"
        if embedding_model is None:
            embedding_model = config.topic_modeler_embedding_model if config else "text-embedding-3-small"
        if embedding_batch_size is None:
            embedding_batch_size = config.topic_modeler_embedding_batch_size if config else 100
        if n_neighbors is None:
            n_neighbors = config.topic_modeler_umap_n_neighbors if config else 15
        if n_components is None:
            n_components = config.topic_modeler_umap_n_components if config else 5
        if min_dist is None:
            min_dist = config.topic_modeler_umap_min_dist if config else 0.0
        if metric is None:
            metric = config.topic_modeler_umap_metric if config else 'cosine'
        if min_cluster_size is None:
            min_cluster_size = config.topic_modeler_hdbscan_min_cluster_size if config else 15
        if min_samples is None:
            min_samples = config.topic_modeler_hdbscan_min_samples if config else 10
        if cluster_selection_method is None:
            cluster_selection_method = config.topic_modeler_hdbscan_cluster_selection_method if config else 'eom'
        if n_jobs is None:
            n_jobs = config.topic_modeler_computing_n_jobs if config else -1
        if low_memory is None:
            low_memory = config.topic_modeler_computing_low_memory if config else False
        if verbose is None:
            verbose = config.topic_modeler_computing_verbose if config else False

        self.models_path = Path(models_path)
        self.results_path = Path(results_path)
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

        # UMAP參數
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.metric = metric

        # HDBSCAN參數
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method

        # 並行計算參數
        self.n_jobs = n_jobs
        self.low_memory = low_memory
        self.verbose = verbose

        # 建立目錄
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)

        # OpenAI客戶端
        self.client = OpenAI(api_key=self.openai_api_key)

        logger.info("TopicModeler初始化完成")

    def generate_embeddings(
        self,
        texts: List[str],
        cache_name: str = "embeddings"
    ) -> np.ndarray:
        """
        生成文本嵌入向量（帶緩存）

        參數:
            texts: 文本列表
            cache_name: 緩存文件名

        返回:
            嵌入向量numpy陣列
        """
        cache_path = self.models_path / f"{cache_name}.npy"
        index_path = self.models_path / f"{cache_name}_index.json"

        # 檢查緩存
        if cache_path.exists() and index_path.exists():
            logger.info("檢查嵌入緩存...")

            with open(index_path, 'r', encoding='utf-8') as f:
                cache_info = json.load(f)

            # 驗證緩存是否有效
            if (cache_info.get('model') == self.embedding_model and
                cache_info.get('count') == len(texts)):
                logger.info("使用緩存的嵌入向量")
                embeddings = np.load(cache_path)
                return embeddings

        # 生成嵌入
        logger.info(f"生成嵌入向量: {len(texts)} 個文本")
        embeddings_list = []

        for i in tqdm(range(0, len(texts), self.embedding_batch_size), desc="生成嵌入"):
            batch = texts[i:i + self.embedding_batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings_list.extend(batch_embeddings)

                # API速率限制延遲
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"生成嵌入時出錯: {e}")
                raise

        embeddings = np.array(embeddings_list)

        # 保存緩存
        np.save(cache_path, embeddings)

        cache_info = {
            'model': self.embedding_model,
            'count': len(texts),
            'shape': embeddings.shape
        }

        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(cache_info, f, ensure_ascii=False, indent=2)

        logger.info(f"嵌入向量已緩存至: {cache_path}")

        return embeddings

    def train_initial_model(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        model_name: str = "phase2_bertopic_model"
    ) -> Tuple[BERTopic, pd.DataFrame, pd.DataFrame, Optional[np.ndarray]]:
        """
        訓練初始BERTopic模型

        參數:
            texts: 文本列表
            embeddings: 嵌入向量
            model_name: 模型名稱

        返回:
            (BERTopic模型, 主題資訊DataFrame, 帶主題標籤的語料庫DataFrame, 主題機率矩陣)
        """
        logger.info("開始訓練BERTopic模型...")

        # 配置UMAP（啟用並行計算）
        umap_model = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=42,
            n_jobs=self.n_jobs,  # 啟用多線程
            low_memory=self.low_memory,
            verbose=self.verbose
        )

        # 配置HDBSCAN（啟用並行計算）
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True,
            core_dist_n_jobs=self.n_jobs  # 啟用多核心計算
        )

        # 配置CountVectorizer with custom stop words
        # 從配置讀取自定義停用詞列表
        config = _get_config()
        custom_stop_words = []

        if config:
            custom_stop_words = config.topic_modeler_vectorizer_custom_stop_words
            max_features = config.topic_modeler_vectorizer_max_features
            ngram_range = config.topic_modeler_vectorizer_ngram_range
        else:
            # 預設值
            custom_stop_words = [
                'cisco', 'marriott', 'kpmg', 'clorox', 'general motors', 'philip morris',
                'sysco', 'ebay', 'gm', 'morris', 'firm', 'report', 'company', 'business',
                'services', 'products', 'industry', 'employees', 'workforce', 'safety',
                'training', 'compliance', 'program', '2017', '2018', '2019', '2020',
                '2021', '2022', 'asia', 'europe', 'america', 'pacific', 'latin america',
                'latin', 'united states', 'uk', 'us'
            ]
            max_features = 1000
            ngram_range = (1, 2)

        # 合併英文停用詞和自定義停用詞
        from sklearn.feature_extraction import text
        english_stop_words = text.ENGLISH_STOP_WORDS
        all_stop_words = list(english_stop_words.union(set(custom_stop_words)))

        vectorizer_model = CountVectorizer(
            max_features=max_features,
            stop_words=all_stop_words,
            ngram_range=ngram_range
        )

        # 初始化BERTopic
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            top_n_words=10,
            verbose=True,
            calculate_probabilities=True
        )

        # 訓練模型
        topics, probs = topic_model.fit_transform(texts, embeddings)

        logger.info(f"模型訓練完成! 發現 {len(set(topics))} 個主題")

        # 計算離群率
        outlier_count = sum(1 for t in topics if t == -1)
        outlier_rate = outlier_count / len(topics) * 100
        logger.info(f"離群率: {outlier_rate:.2f}%")

        # 保存模型
        model_path = self.models_path / model_name
        topic_model.save(str(model_path), serialization="safetensors")
        logger.info(f"模型已保存至: {model_path}")

        # 獲取主題資訊
        topic_info = topic_model.get_topic_info()
        topic_info_path = self.results_path / "phase2_topics.csv"
        topic_info.to_csv(topic_info_path, index=False, encoding='utf-8')
        logger.info(f"主題資訊已保存至: {topic_info_path}")

        # 建立帶主題標籤的語料庫
        corpus_with_topics = pd.DataFrame({
            'text': texts,
            'topic': topics
        })
        corpus_path = self.results_path / "phase2_corpus_with_topics.csv"
        corpus_with_topics.to_csv(corpus_path, index=False, encoding='utf-8')
        logger.info(f"帶主題標籤的語料庫已保存至: {corpus_path}")

        # 保存主題機率矩陣
        if probs is not None:
            probs_path = self.models_path / "phase2_doc_topic_probs.npy"
            np.save(probs_path, probs)
            logger.info(f"主題機率矩陣已保存至: {probs_path}")

        return topic_model, topic_info, corpus_with_topics, probs

    def get_topic_summary(self, topic_model: BERTopic) -> Dict:
        """
        獲取主題摘要（JSON格式）

        參數:
            topic_model: BERTopic模型

        返回:
            主題摘要字典
        """
        topic_info = topic_model.get_topic_info()

        summary = {
            'metadata': {
                'total_topics': len(topic_info),
                'embedding_model': self.embedding_model,
                'umap_params': {
                    'n_neighbors': self.n_neighbors,
                    'n_components': self.n_components,
                    'min_dist': self.min_dist,
                    'metric': self.metric
                },
                'hdbscan_params': {
                    'min_cluster_size': self.min_cluster_size,
                    'min_samples': self.min_samples,
                    'cluster_selection_method': self.cluster_selection_method
                }
            },
            'topics': []
        }

        # 遍歷每個主題
        for _, row in topic_info.iterrows():
            topic_id = int(row['Topic'])

            # 獲取主題關鍵詞
            topic_words = topic_model.get_topic(topic_id)

            if topic_words:
                keywords = [{'word': word, 'score': float(score)} for word, score in topic_words]
            else:
                keywords = []

            topic_dict = {
                'topic_id': topic_id,
                'count': int(row['Count']),
                'name': row.get('Name', f'Topic_{topic_id}'),
                'keywords': keywords
            }

            summary['topics'].append(topic_dict)

        return summary

    def export_topic_summary_json(
        self,
        topic_model: BERTopic,
        output_filename: str = "phase2_topic_summary.json"
    ) -> str:
        """
        導出主題摘要為JSON文件

        參數:
            topic_model: BERTopic模型
            output_filename: 輸出文件名

        返回:
            輸出文件路徑
        """
        logger.info("生成主題摘要JSON...")

        summary = self.get_topic_summary(topic_model)

        output_path = self.results_path / output_filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"主題摘要已導出至: {output_path}")

        return str(output_path)


# 便捷函數
def load_topic_model(model_path: str = "./data/models/phase2_bertopic_model") -> BERTopic:
    """
    載入BERTopic模型

    參數:
        model_path: 模型路徑

    返回:
        BERTopic模型
    """
    logger.info(f"載入BERTopic模型: {model_path}")
    topic_model = BERTopic.load(model_path)
    logger.info("模型載入完成")
    return topic_model

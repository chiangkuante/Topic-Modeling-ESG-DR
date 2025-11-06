#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試配置系統是否正常運作
"""
import sys

def test_config_loading():
    """測試配置載入"""
    print("=" * 50)
    print("測試配置載入...")
    print("=" * 50)

    try:
        from src.config_loader import get_config, load_keywords
        config = get_config()

        print("\n✓ 配置載入成功")
        print(f"  配置文件路徑: {config.config_path}")

        # 測試路徑配置
        print("\n路徑配置:")
        print(f"  root: {config.root}")
        print(f"  raw_data_path: {config.raw_data_path}")
        print(f"  processed_data_path: {config.processed_data_path}")
        print(f"  models_path: {config.models_path}")
        print(f"  results_path: {config.results_path}")

        # 測試 Pipeline 配置
        print("\nPipeline 配置:")
        print(f"  semantic_chunking: {config.semantic_chunking}")
        print(f"  crawl: {config.crawl}")

        # 測試 Crawler 配置
        print("\nCrawler 配置:")
        print(f"  start_year: {config.crawler_start_year}")
        print(f"  end_year: {config.crawler_end_year}")
        print(f"  max_results: {config.crawler_max_results}")
        print(f"  throttle_sec: {config.crawler_throttle_sec}")

        # 測試 Data Loader 配置
        print("\nData Loader 配置:")
        print(f"  min_sentence_length: {config.data_loader_min_sentence_length}")
        print(f"  max_chunk_tokens: {config.data_loader_max_chunk_tokens}")
        print(f"  boilerplate_keywords: {len(config.data_loader_boilerplate_keywords)} 個關鍵詞")

        # 測試 Topic Modeler 配置
        print("\nTopic Modeler 配置:")
        print(f"  embedding_model: {config.topic_modeler_embedding_model}")
        print(f"  embedding_batch_size: {config.topic_modeler_embedding_batch_size}")
        print(f"  UMAP n_neighbors: {config.topic_modeler_umap_n_neighbors}")
        print(f"  UMAP n_components: {config.topic_modeler_umap_n_components}")
        print(f"  HDBSCAN min_cluster_size: {config.topic_modeler_hdbscan_min_cluster_size}")
        print(f"  HDBSCAN min_samples: {config.topic_modeler_hdbscan_min_samples}")
        print(f"  Computing n_jobs: {config.topic_modeler_computing_n_jobs}")

        # 測試關鍵詞載入
        print("\n載入關鍵詞配置...")
        keywords = load_keywords(config.root)
        report_keywords = keywords.get("report_keywords", [])
        print(f"✓ 關鍵詞載入成功: {len(report_keywords)} 個報告關鍵詞")
        for kw in report_keywords:
            print(f"  - {kw}")

        print("\n" + "=" * 50)
        print("所有配置測試通過!")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"\n✗ 配置載入失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_class_initialization():
    """測試類別初始化"""
    print("\n" + "=" * 50)
    print("測試類別初始化...")
    print("=" * 50)

    try:
        # 測試 DataLoader
        print("\n測試 DataLoader 初始化...")
        from src.data_loader import DataLoader
        data_loader = DataLoader()
        print(f"✓ DataLoader 初始化成功")
        print(f"  raw_data_path: {data_loader.raw_data_path}")
        print(f"  min_sentence_length: {data_loader.min_sentence_length}")
        print(f"  max_chunk_tokens: {data_loader.max_chunk_tokens}")

        # 測試 TopicModeler（需要 API key，只測試初始化不測試實際功能）
        print("\n測試 TopicModeler 初始化...")
        # 跳過 TopicModeler 測試，因為需要 OpenAI API key
        print("  (跳過 - 需要 OpenAI API key)")

        print("\n" + "=" * 50)
        print("類別初始化測試通過!")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"\n✗ 類別初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_config_loading()

    if success:
        success = test_class_initialization()

    if success:
        print("\n" + "=" * 50)
        print("✓ 所有測試通過！配置系統運作正常。")
        print("=" * 50)
        sys.exit(0)
    else:
        print("\n" + "=" * 50)
        print("✗ 測試失敗")
        print("=" * 50)
        sys.exit(1)

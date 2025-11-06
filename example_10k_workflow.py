#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_10k_workflow.py
示範如何使用 data_loader_10k.py 和 topic_modeler.py 處理 10-K 報告
"""
import sys
from pathlib import Path

# 添加 src 目錄到路徑
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader_10k import DataLoader10K, load_corpus_10k
from src.topic_modeler import TopicModeler

def main():
    print("=" * 80)
    print("10-K 報告主題建模工作流程範例")
    print("=" * 80)

    # ========== 步驟 1: 探索資料集 ==========
    print("\n步驟 1: 探索 10-K 資料集...")
    print("-" * 80)

    loader = DataLoader10K(
        raw_data_path="./data/sec-edgar-filings",
        processed_data_path="./data/processed_corpus",
        min_sentence_length=50,
        max_chunk_tokens=512
    )

    stats = loader.explore_dataset()
    print(f"找到 {stats['total_companies']} 家公司")
    print(f"總共 {stats['total_files']} 個 10-K 文件")
    print(f"提交年份: {stats['filing_years']}")

    # 顯示前10家公司
    print("\n前 10 家公司:")
    for i, (ticker, count) in enumerate(list(stats['companies'].items())[:10]):
        print(f"  {i+1}. {ticker}: {count} 個文件")

    # ========== 步驟 2: 建立語料庫 ==========
    print("\n步驟 2: 建立 10-K 語料庫...")
    print("-" * 80)
    print("注意: 這可能需要一些時間，取決於文件數量")
    print("提示: 可以使用 tickers 參數只處理特定公司")

    # 範例: 只處理前5家公司（快速測試）
    # sample_tickers = list(stats['companies'].keys())[:5]
    # corpus_df = loader.build_corpus(
    #     output_filename="corpus_10k_sample.csv",
    #     use_semantic_chunking=False,  # 設為 True 以使用語義分塊（需要 OpenAI API）
    #     tickers=sample_tickers
    # )

    # 完整處理（取消註解以執行）
    # corpus_df = loader.build_corpus(
    #     output_filename="corpus_10k.csv",
    #     use_semantic_chunking=False
    # )

    print("建立語料庫已跳過（取消註解以執行）")

    # ========== 步驟 3: 載入語料庫 ==========
    print("\n步驟 3: 載入已處理的語料庫...")
    print("-" * 80)

    # 假設已經建立了語料庫
    corpus_path = "./data/processed_corpus/corpus_10k.csv"
    if Path(corpus_path).exists():
        corpus_df = load_corpus_10k(corpus_path)
        print(f"語料庫資訊:")
        print(f"  總文本塊數: {len(corpus_df)}")
        print(f"  欄位: {list(corpus_df.columns)}")
        print(f"\n前 3 筆資料:")
        print(corpus_df.head(3))
    else:
        print(f"語料庫文件不存在: {corpus_path}")
        print("請先執行步驟 2 建立語料庫")
        corpus_df = None

    # ========== 步驟 4: 主題建模 ==========
    if corpus_df is not None and len(corpus_df) > 0:
        print("\n步驟 4: 執行主題建模...")
        print("-" * 80)

        # 初始化 TopicModeler
        topic_modeler = TopicModeler(
            models_path="./data/models",
            results_path="./data/results",
            embedding_model="text-embedding-3-small",
            embedding_batch_size=100,
            # UMAP 參數
            n_neighbors=15,
            n_components=5,
            # HDBSCAN 參數
            min_cluster_size=15,
            min_samples=10
        )

        # 準備文本列表
        texts = corpus_df['text'].tolist()
        print(f"準備進行主題建模: {len(texts)} 個文本塊")

        # 生成嵌入向量
        print("生成嵌入向量...")
        # embeddings = topic_modeler.generate_embeddings(
        #     texts,
        #     cache_name="embeddings_10k"
        # )

        # 訓練模型
        # print("訓練 BERTopic 模型...")
        # topic_model, topic_info, corpus_with_topics, probs = topic_modeler.train_initial_model(
        #     texts,
        #     embeddings,
        #     model_name="bertopic_10k_model"
        # )

        # 導出主題摘要
        # topic_modeler.export_topic_summary_json(
        #     topic_model,
        #     output_filename="topic_summary_10k.json"
        # )

        print("主題建模已跳過（取消註解以執行）")
        print("\n提示: 取消註解上述代碼以執行完整的主題建模流程")

    print("\n" + "=" * 80)
    print("工作流程範例完成!")
    print("=" * 80)

    print("\n快速開始:")
    print("1. 探索資料集: loader.explore_dataset()")
    print("2. 建立語料庫: loader.build_corpus()")
    print("3. 載入語料庫: load_corpus_10k('path/to/corpus.csv')")
    print("4. 主題建模: 使用 TopicModeler 類別")
    print("\n詳細使用方法請參閱 data_loader_10k.py 和 topic_modeler.py")


if __name__ == "__main__":
    main()

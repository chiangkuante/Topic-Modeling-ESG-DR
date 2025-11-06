#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_test_10k.py
快速測試 data_loader_10k.py - 處理少量公司的 10-K 報告
"""
import sys
from pathlib import Path

# 添加 src 目錄到路徑
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader_10k import DataLoader10K

def main():
    print("=" * 80)
    print("10-K Data Loader 快速測試")
    print("=" * 80)

    # 初始化載入器
    loader = DataLoader10K(
        raw_data_path="./data/sec-edgar-filings",
        processed_data_path="./data/processed_corpus",
        min_sentence_length=50,
        max_chunk_tokens=512
    )

    # 探索資料集
    print("\n正在探索資料集...")
    stats = loader.explore_dataset()

    print(f"\n資料集統計:")
    print(f"  公司數量: {stats['total_companies']}")
    print(f"  10-K 文件總數: {stats['total_files']}")
    print(f"  年份範圍: {stats['filing_years']}")

    # 選擇前 3 家公司進行測試
    sample_tickers = list(stats['companies'].keys())[:3]
    print(f"\n選擇以下公司進行快速測試: {', '.join(sample_tickers)}")

    # 建立語料庫（只處理選定的公司）
    print(f"\n開始處理 {len(sample_tickers)} 家公司的 10-K 報告...")
    print("這可能需要幾分鐘時間...\n")

    corpus_df = loader.build_corpus(
        output_filename="corpus_10k_quick_test.csv",
        use_semantic_chunking=False,  # 使用簡單分塊（快速）
        tickers=sample_tickers
    )

    # 顯示結果
    print("\n" + "=" * 80)
    print("處理完成！")
    print("=" * 80)

    print(f"\n語料庫資訊:")
    print(f"  總文本塊數: {len(corpus_df)}")
    print(f"  欄位: {list(corpus_df.columns)}")

    print(f"\n按公司統計:")
    for ticker in sample_tickers:
        count = len(corpus_df[corpus_df['ticker'] == ticker])
        print(f"  {ticker}: {count} 個文本塊")

    print(f"\n按年份統計:")
    year_counts = corpus_df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} 個文本塊")

    # 顯示範例文本
    print(f"\n範例文本塊:")
    print("-" * 80)
    sample_text = corpus_df.iloc[0]['text']
    print(f"公司: {corpus_df.iloc[0]['ticker']}")
    print(f"年份: {corpus_df.iloc[0]['year']}")
    print(f"文本長度: {len(sample_text)} 字符")
    print(f"\n文本預覽:")
    print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
    print("-" * 80)

    print(f"\n語料庫已保存至: ./data/processed_corpus/corpus_10k_quick_test.csv")
    print("\n下一步:")
    print("1. 檢查生成的 CSV 文件")
    print("2. 使用 topic_modeler.py 進行主題建模")
    print("3. 參閱 DATA_LOADER_10K_README.md 了解更多用法")


if __name__ == "__main__":
    main()

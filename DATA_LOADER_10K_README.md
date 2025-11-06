# Data Loader 10-K 使用指南

`data_loader_10k.py` 是專門用於從 SEC Edgar 10-K 報告中提取和處理資料的模組。

## 功能特點

- 從 `sec-edgar-filings` 目錄讀取 10-K 報告
- 自動清理 HTML/XML 標記
- 過濾樣板化內容和無效句子
- 支援簡單分塊和語義分塊
- 與 `topic_modeler.py` 完全兼容
- 生成包含元數據的結構化語料庫

## 資料集統計

根據當前的 `sec-edgar-filings` 目錄：
- **公司數量**: 442 家
- **10-K 文件總數**: 4,420 個
- **年份範圍**: 2014-2024

## 快速開始

### 1. 基本使用

```python
from src.data_loader_10k import DataLoader10K, load_corpus_10k

# 初始化載入器
loader = DataLoader10K(
    raw_data_path="./data/sec-edgar-filings",
    processed_data_path="./data/processed_corpus",
    min_sentence_length=50,
    max_chunk_tokens=512
)

# 探索資料集
stats = loader.explore_dataset()
print(f"找到 {stats['total_companies']} 家公司")
print(f"總共 {stats['total_files']} 個 10-K 文件")
```

### 2. 建立語料庫

```python
# 處理所有公司
corpus_df = loader.build_corpus(
    output_filename="corpus_10k.csv",
    use_semantic_chunking=False  # True 需要 OpenAI API
)

# 只處理特定公司（快速測試）
corpus_df = loader.build_corpus(
    output_filename="corpus_10k_sample.csv",
    use_semantic_chunking=False,
    tickers=['AAPL', 'MSFT', 'GOOGL']
)
```

### 3. 載入已處理的語料庫

```python
# 載入語料庫
corpus_df = load_corpus_10k("./data/processed_corpus/corpus_10k.csv")

print(f"文本塊數: {len(corpus_df)}")
print(f"欄位: {list(corpus_df.columns)}")
# 欄位: ['ticker', 'year', 'text', 'source_file', 'filing_type']
```

### 4. 與 TopicModeler 整合

```python
from src.topic_modeler import TopicModeler

# 初始化主題建模器
topic_modeler = TopicModeler(
    models_path="./data/models",
    results_path="./data/results"
)

# 準備文本
texts = corpus_df['text'].tolist()

# 生成嵌入向量
embeddings = topic_modeler.generate_embeddings(
    texts,
    cache_name="embeddings_10k"
)

# 訓練模型
topic_model, topic_info, corpus_with_topics, probs = topic_modeler.train_initial_model(
    texts,
    embeddings,
    model_name="bertopic_10k_model"
)

# 導出結果
topic_modeler.export_topic_summary_json(
    topic_model,
    output_filename="topic_summary_10k.json"
)
```

## 參數說明

### DataLoader10K 初始化參數

- `raw_data_path`: 10-K 文件的根目錄（預設: `./data/sec-edgar-filings`）
- `processed_data_path`: 處理後語料庫的保存路徑（預設: `./data/processed_corpus`）
- `min_sentence_length`: 句子最小長度，過濾太短的句子（預設: 50）
- `max_chunk_tokens`: 每個文本塊的最大 token 數（預設: 512）
- `boilerplate_keywords`: 樣板化關鍵詞列表，用於過濾標準格式內容
- `openai_api_key`: OpenAI API 金鑰（用於語義分塊，可選）

### build_corpus 參數

- `output_filename`: 輸出的 CSV 檔名（預設: `corpus_10k.csv`）
- `use_semantic_chunking`: 是否使用 LLM 進行語義分塊（預設: False）
- `tickers`: 要處理的公司代碼列表（預設: None，處理全部）

## 資料結構

### 輸入資料結構

```
sec-edgar-filings/
├── AAPL/
│   └── 10-K/
│       ├── 0000320193-14-000019/
│       │   └── full-submission.txt
│       ├── 0000320193-15-000010/
│       │   └── full-submission.txt
│       └── ...
├── MSFT/
│   └── 10-K/
│       └── ...
└── ...
```

### 輸出語料庫格式

CSV 檔案包含以下欄位：
- `ticker`: 公司代碼（如 AAPL）
- `year`: 報告年份（如 2023）
- `text`: 處理後的文本塊
- `source_file`: 原始文件名
- `filing_type`: 報告類型（10-K）

## 文本處理流程

1. **提取文本**: 從 `full-submission.txt` 提取內容
2. **清理標記**: 移除 HTML/XML/XBRL 標記
3. **文本清理**: 移除多餘空白和特殊字符
4. **句子分割**: 使用 NLTK 進行句子分割
5. **句子過濾**:
   - 移除太短的句子
   - 過濾樣板化內容
   - 移除包含過多數字的句子（表格數據）
6. **文本分塊**:
   - 簡單分塊：按 token 數分割
   - 語義分塊：使用 LLM 進行語義相關的分塊

## 範例腳本

運行完整的工作流程範例：

```bash
# 使用虛擬環境
.venv/bin/python example_10k_workflow.py
```

## 注意事項

1. **處理時間**: 處理全部 4,420 個文件可能需要較長時間，建議先用小樣本測試
2. **記憶體使用**: 大量文本處理需要足夠的記憶體
3. **語義分塊**: 需要 OpenAI API 金鑰，且會產生 API 費用
4. **依賴套件**: 需要安裝 `beautifulsoup4` 和 `lxml`

## 與 data_loader.py 的比較

| 特性 | data_loader.py | data_loader_10k.py |
|------|---------------|-------------------|
| 輸入格式 | PDF 文件 | TXT 文件（10-K 提交） |
| 目錄結構 | TICKER/YEAR/*.pdf | TICKER/10-K/ACCESSION/*.txt |
| 文本提取 | pdfplumber/PyPDF2 | BeautifulSoup (HTML 解析) |
| 元數據 | ticker, year, source_file | ticker, year, source_file, filing_type |
| 特殊處理 | 無 | 移除 XBRL 標記、過濾表格數據 |

## 疑難排解

### 問題：ModuleNotFoundError: No module named 'bs4'

```bash
# 使用 uv 安裝依賴
uv pip install beautifulsoup4 lxml
```

### 問題：提取的文本包含過多格式內容

調整 `boilerplate_keywords` 參數添加更多過濾詞：

```python
loader = DataLoader10K(
    boilerplate_keywords=[
        "forward-looking statement",
        "safe harbor",
        # 添加更多自定義過濾詞
    ]
)
```

### 問題：文本塊太大或太小

調整 `max_chunk_tokens` 參數：

```python
loader = DataLoader10K(
    max_chunk_tokens=256  # 或更大的值如 1024
)
```

## 進階使用

### 批次處理特定年份

```python
# 只處理 2020-2023 年的報告
# （需要自行實現年份過濾邏輯）
```

### 自定義文本清理

繼承 `DataLoader10K` 類別並覆寫 `clean_text` 方法：

```python
class CustomDataLoader10K(DataLoader10K):
    def clean_text(self, text: str) -> str:
        # 自定義清理邏輯
        text = super().clean_text(text)
        # 添加額外的清理步驟
        return text
```

## 相關文件

- `data_loader.py`: PDF 文件載入器
- `topic_modeler.py`: 主題建模模組
- `example_10k_workflow.py`: 完整工作流程範例
- `config/config.yaml`: 配置文件

## 授權

與主專案相同

## 貢獻

歡迎提交 Issue 和 Pull Request！

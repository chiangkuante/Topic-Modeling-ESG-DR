# ESG報告主題建模與數位韌性量化框架

自動化框架，用於爬取企業ESG報告、進行主題建模、映射至數位韌性構面，並產出量化的數位韌性分數。

## 功能特點

- ESG報告自動爬取（使用Brave Search API）
- PDF文本提取與預處理
- 語義分塊（可選，使用LLM）
- 基於BERTopic的主題建模
- 主題映射至數位韌性構面
- 數位韌性指數(DRI)計算

## 技術棧

- Python 3.12+
- UV (套件管理)
- Pandas, NumPy
- BERTopic, UMAP, HDBSCAN
- OpenAI API
- Scikit-learn, NLTK

## 專案結構

```
project_root/
├── data/                    # 數據目錄
│   ├── raw/                # 原始PDF報告
│   ├── metadata/           # 爬蟲manifest
│   ├── processed_corpus/   # 處理後的語料庫
│   ├── models/             # 儲存的模型
│   └── results/            # 最終結果
├── src/                    # 原始碼
│   ├── esg_crawler.py     # ESG報告爬蟲
│   ├── data_loader.py     # 資料載入與預處理
│   ├── topic_modeler.py   # 主題建模模組
│   ├── mapper.py          # 主題映射模組
│   └── scorer.py          # 分數計算模組
├── notebooks/              # Jupyter Notebooks
│   └── main.ipynb         # 主要互動式筆記本
├── tests/                  # 測試程式碼
├── .env                    # 環境變數（需自行建立）
├── .env.example            # 環境變數範例
├── pyproject.toml          # 專案配置
└── main.py                 # 主執行腳本
```

## 安裝步驟

### 1. 安裝UV套件管理器

```bash
pip install uv
```

### 2. 建立虛擬環境

```bash
uv venv .venv
```

### 3. 啟動虛擬環境

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 4. 安裝依賴套件

```bash
uv pip install -e .
```

### 5. 設定環境變數

複製 `.env.example` 為 `.env` 並填入API金鑰：

```bash
cp .env.example .env
```

編輯 `.env` 檔案：

```
BRAVE_API_KEY=your_brave_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

## 使用方式

### 使用命令行介面

**執行所有階段:**
```bash
uv run main.py --stage all
```

**執行特定階段:**
```bash
# 階段0: 環境設定
uv run main.py --stage 0

# 階段1: 資料載入與預處理
uv run main.py --stage 1

# 階段2: 主題建模
uv run main.py --stage 2
```

### 階段0: 爬取 ESG 報告（可選）

使用 Brave Search API 自動爬取 S&P 500 公司的 ESG 報告：

```bash
# 爬取 2017-2018 年的報告
uv run main.py --stage 0 --crawl --start-year 2021 --end-year 2024

# 爬取 2017-2024 年的報告（自訂參數）
uv run main.py --stage 0 --crawl --start-year 2017 --end-year 2024 --max-results 3 --throttle-sec 1.1

# 自訂爬蟲參數
uv run main.py --stage 0 --crawl \
  --start-year 2019 \
  --end-year 2021 \
  --max-results 10 \
  --throttle-sec 1.5
```

**爬蟲參數說明:**
- `--crawl`: 啟用爬蟲功能
- `--start-year`: 起始年份（預設: 2017）
- `--end-year`: 結束年份（預設: 2018）
- `--max-results`: 每次搜尋的最大結果數（預設: 8）
- `--throttle-sec`: API 請求間隔秒數（預設: 1.0）

**前置需求:**
1. 在 `.env` 文件中設定 `BRAVE_API_KEY`
2. 確保 `data/sp500_2017-01-27.csv` 文件存在

**輸出:**
- PDF 文件: `data/raw/{ticker}/{year}/{filename}.pdf`
- 下載清單: `data/metadata/esg_manifest.csv`

### 使用語義分塊

```bash
uv run main.py --stage 1 --semantic-chunking
```

### 使用Jupyter Notebook

啟動Jupyter Notebook：

```bash
jupyter notebook notebooks/main.ipynb
```

在notebook中可以互動式地執行各個階段。

## 完整命令行參數

### 主要參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--stage` | str | `all` | 執行階段: `0`, `1`, `2`, `all` |
| `--semantic-chunking` | flag | False | 使用語義分塊（需要 OpenAI API） |

### 爬蟲相關參數（Stage 0）

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--crawl` | flag | False | 啟用 Brave Search API 爬蟲 |
| `--start-year` | int | 2017 | 爬蟲起始年份 |
| `--end-year` | int | 2018 | 爬蟲結束年份 |
| `--max-results` | int | 8 | 每次搜尋的最大結果數 |
| `--throttle-sec` | float | 1.0 | API 請求間隔秒數 |

### 使用範例

```bash
# 1. 完整流程（不包含爬蟲）
uv run main.py --stage all

# 2. 只執行爬蟲
uv run main.py --stage 0 --crawl --start-year 2020 --end-year 2022

# 3. 爬蟲 + 資料處理 + 主題建模
uv run main.py --stage 0 --crawl --start-year 2020 --end-year 2020
uv run main.py --stage 1
uv run main.py --stage 2

# 4. 使用語義分塊進行資料處理
uv run main.py --stage 1 --semantic-chunking

# 5. 只重新訓練主題模型（使用緩存的嵌入向量）
uv run main.py --stage 2
```

## 階段說明

### 階段0: 環境設定
- 檢查目錄結構
- 驗證環境變數

**階段0（可選）: ESG 報告爬蟲**
- 使用 Brave Search API 搜尋 ESG 報告
- 下載 PDF 文件
- 生成下載清單（manifest）
- 自動組織文件結構

### 階段1: 資料載入與預處理
- 探索資料集結構
- 提取PDF文本
- 文本清理與過濾
- 文本分塊（支援語義分塊）
- 建立語料庫CSV

**輸出:**
- `data/processed_corpus/corpus.csv`

### 階段2: 主題建模
- 生成文本嵌入向量（使用OpenAI）
- 訓練BERTopic模型（UMAP + HDBSCAN）
- 生成主題資訊
- 導出JSON摘要

**輸出:**
- `data/models/phase2_bertopic_model/` - BERTopic模型
- `data/models/embeddings.npy` - 嵌入向量緩存
- `data/results/phase2_topics.csv` - 主題資訊
- `data/results/phase2_corpus_with_topics.csv` - 帶主題標籤的語料庫
- `data/results/phase2_topic_summary.json` - 主題摘要JSON

## 階段2結果格式（JSON）

```json
{
  "metadata": {
    "total_topics": 25,
    "embedding_model": "text-embedding-3-small",
    "umap_params": {
      "n_neighbors": 15,
      "n_components": 5,
      "min_dist": 0.0,
      "metric": "cosine"
    },
    "hdbscan_params": {
      "min_cluster_size": 15,
      "min_samples": 10,
      "cluster_selection_method": "eom"
    }
  },
  "topics": [
    {
      "topic_id": 0,
      "count": 150,
      "name": "0_climate_carbon_emissions_reduction",
      "keywords": [
        {"word": "climate", "score": 0.85},
        {"word": "carbon", "score": 0.82},
        {"word": "emissions", "score": 0.79}
      ]
    }
  ]
}
```

## 配置參數

### 資料載入參數

- `min_sentence_length`: 句子最小長度（預設：50）
- `max_chunk_tokens`: 最大chunk token數（預設：512）
- `boilerplate_keywords`: 樣板化關鍵詞列表

### 主題建模參數

**嵌入參數:**
- `embedding_model`: 嵌入模型名稱（預設：text-embedding-3-small）
- `embedding_batch_size`: 嵌入批次大小（預設：100）

**UMAP參數:**
- `n_neighbors`: 鄰居數（預設：15）
- `n_components`: 維度數（預設：5）
- `min_dist`: 最小距離（預設：0.0）
- `metric`: 距離度量（預設：cosine）

**HDBSCAN參數:**
- `min_cluster_size`: 最小聚類大小（預設：15）
- `min_samples`: 最小樣本數（預設：10）
- `cluster_selection_method`: 聚類選擇方法（預設：eom）

## 日誌

執行過程中的日誌會同時輸出到：
- 終端（stdout）
- `esg_pipeline.log` 文件

## 測試

執行測試：

```bash
pytest tests/
```

## 注意事項

1. 確保 `.env` 文件中的API金鑰正確設定
2. 首次執行階段2時會生成嵌入向量，需要一定時間
3. 嵌入向量會被緩存，後續執行會更快
4. 語義分塊需要OpenAI API且會增加成本
5. 確保 `data/raw/` 中有足夠的PDF文件

## 下一步開發

- 階段3: 主題映射模組
- 階段4: 分數計算與DRI
- 階段5: 完整流程整合
- 階段6: 單元測試與整合測試
- 階段7: Web介面（Streamlit）

## 授權

MIT License

## 聯絡方式

如有問題請提交issue或聯絡開發團隊。

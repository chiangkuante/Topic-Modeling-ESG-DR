# 配置系統遷移說明

## 概述

所有檔案已成功修改，除了 `--stage` 參數外，其他 CLI 參數都已改為使用 `config/config.yaml` 進行配置。

## 變更摘要

### 新增檔案

1. **config/config.yaml** - 主要配置檔案，包含所有參數設定
2. **config/keywords.yaml** - ESG 報告關鍵詞配置
3. **src/config_loader.py** - 配置載入模組

### 修改檔案

1. **main.py**
   - 移除所有 CLI 參數（除了 `--stage`）
   - 從 config 讀取所有設定
   - 使用方式：`python main.py --stage [0|1|2|all]`

2. **src/esg_crawler.py**
   - 移除 `parse_args()` 函數
   - 新增 `create_args_from_config()` 函數從配置讀取參數
   - 可以獨立執行：`python -m src.esg_crawler`

3. **src/data_loader.py**
   - `DataLoader` 類別的所有參數改為可選（Optional）
   - 未提供參數時，自動從 config 讀取預設值
   - 向後相容，仍可手動傳入參數

4. **src/topic_modeler.py**
   - `TopicModeler` 類別的所有參數改為可選（Optional）
   - 未提供參數時，自動從 config 讀取預設值
   - 向後相容，仍可手動傳入參數

## 使用方式

### 1. 基本使用

執行完整 pipeline：

```bash
python main.py
```

或指定階段：

```bash
python main.py --stage 0  # 環境設定
python main.py --stage 1  # 資料載入
python main.py --stage 2  # 主題建模
```

### 2. 修改配置

編輯 `config/config.yaml` 檔案來調整參數：

```yaml
# 修改爬蟲年份範圍
crawler:
  start_year: 2019
  end_year: 2022

# 調整 Topic Modeler 參數
topic_modeler:
  hdbscan:
    min_cluster_size: 50
    min_samples: 15
```

### 3. 環境變數

以下參數仍需在 `.env` 檔案中設定：

```
OPENAI_API_KEY=your_openai_api_key
BRAVE_API_KEY=your_brave_api_key  # 只有啟用爬蟲時需要
EMBEDDING_MODEL=text-embedding-3-small  # 可選，會覆蓋 config.yaml 設定
LLM_MODEL=gpt-4o-mini  # 可選
```

### 4. 獨立運行爬蟲

```bash
python -m src.esg_crawler
```

爬蟲會自動從 `config/config.yaml` 讀取參數。

## 參數優先順序

1. **手動傳入的參數**（最高優先）
2. **環境變數**（中等優先，僅適用於特定參數如 EMBEDDING_MODEL）
3. **config.yaml 配置**（預設）

## 配置檔案結構

### config/config.yaml

```yaml
paths:           # 路徑設定
  root: "."
  raw_data: "./data/raw"
  processed_data: "./data/processed_corpus"
  sp500_csv: "./data/sp500_2025.csv"  # S&P 500 公司列表，可改為其他年份
  manifest: "./data/metadata/esg_manifest.csv"
  ...

pipeline:        # Pipeline 設定
  semantic_chunking: false
  crawl: false

crawler:         # 爬蟲設定
  start_year: 2017
  end_year: 2018
  ...

data_loader:     # 資料載入器設定
  min_sentence_length: 50
  max_chunk_tokens: 512
  ...

topic_modeler:   # 主題建模器設定
  embedding_model: "text-embedding-3-small"
  umap: { ... }
  hdbscan: { ... }
  vectorizer: { ... }
  bertopic: { ... }
  computing: { ... }
```

### config/keywords.yaml

```yaml
report_keywords:
  - "sustainability report"
  - "ESG report"
  - "corporate responsibility report"
  ...
```

## 測試配置系統

運行測試腳本來驗證配置是否正常：

```bash
python test_config.py
```

預期輸出應顯示：

```
✓ 配置載入成功
✓ 關鍵詞載入成功
✓ 所有配置測試通過!
```

## 向後相容性

所有修改都保持向後相容：

1. **DataLoader** 和 **TopicModeler** 仍可手動傳入參數
2. 未提供參數時才會從 config 讀取
3. 環境變數仍然有效

範例：

```python
from src.data_loader import DataLoader

# 使用 config 預設值
loader1 = DataLoader()

# 手動指定參數（覆蓋 config）
loader2 = DataLoader(
    raw_data_path="/custom/path",
    min_sentence_length=100
)
```

## 故障排除

### 問題：找不到配置檔案

**錯誤訊息：**
```
FileNotFoundError: 配置文件不存在: config/config.yaml
```

**解決方案：**
確認 `config/config.yaml` 存在且路徑正確。

### 問題：YAML 格式錯誤

**錯誤訊息：**
```
ValueError: 配置文件格式錯誤
```

**解決方案：**
檢查 YAML 語法，注意縮排和格式。

### 問題：缺少 pyyaml 套件

**錯誤訊息：**
```
ModuleNotFoundError: No module named 'yaml'
```

**解決方案：**
```bash
pip install pyyaml
```

## 參數對照表

### 移除的 CLI 參數

以下 CLI 參數已移除，請在 `config/config.yaml` 中設定：

| 舊 CLI 參數 | 新配置位置 |
|-----------|----------|
| `--semantic-chunking` | `pipeline.semantic_chunking` |
| `--crawl` | `pipeline.crawl` |
| `--start-year` | `crawler.start_year` |
| `--end-year` | `crawler.end_year` |
| `--max-results` | `crawler.max_results` |
| `--throttle-sec` | `crawler.throttle_sec` |

### 保留的 CLI 參數

| CLI 參數 | 說明 | 範例 |
|---------|------|------|
| `--stage` | 執行階段選擇 | `--stage 0`, `--stage 1`, `--stage 2`, `--stage all` |

## 檔案清單

### 新增檔案
- `config/config.yaml` - 主要配置檔案
- `config/keywords.yaml` - 關鍵詞配置
- `src/config_loader.py` - 配置載入模組
- `test_config.py` - 配置測試腳本
- `CONFIG_MIGRATION.md` - 本文件

### 修改檔案
- `main.py` - 主程式
- `src/esg_crawler.py` - 爬蟲模組
- `src/data_loader.py` - 資料載入器
- `src/topic_modeler.py` - 主題建模器

## 總結

✅ 所有 CLI 參數（除了 `--stage`）已成功遷移至 `config.yaml`
✅ 配置系統測試通過
✅ 保持向後相容性
✅ 程式碼結構更清晰、易維護

現在你可以：
- 輕鬆修改配置而不需要記憶複雜的 CLI 參數
- 版本控制配置檔案
- 為不同環境維護不同的配置檔案

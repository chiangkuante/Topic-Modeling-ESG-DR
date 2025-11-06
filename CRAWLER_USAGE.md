# ESG Crawler 使用指南

## 快速開始

### 使用 uv 執行爬蟲

```bash
# 在專案根目錄執行
uv run python -m src.esg_crawler
```

## 配置說明

### 1. 設定 S&P 500 公司列表

編輯 `config/config.yaml`：

```yaml
paths:
  # 修改 S&P 500 CSV 檔案路徑
  sp500_csv: "./data/sp500_2025.csv"

  # 其他可選路徑：
  # sp500_csv: "./data/sp500_2017-01-27.csv"
  # sp500_csv: "./data/custom_companies.csv"
```

### 2. 設定 Brave API Key

在專案根目錄的 `.env` 檔案中添加：

```bash
BRAVE_API_KEY=your_brave_api_key_here
```

### 3. 調整爬蟲參數

編輯 `config/config.yaml` 的 crawler 區段：

```yaml
crawler:
  # 爬取年份範圍
  start_year: 2017
  end_year: 2018

  # 每個查詢的結果數
  max_results: 20           # 每頁最大結果（<=20）
  max_results_total: 60     # 總結果數

  # 請求間隔設定（避免被封鎖）
  throttle_sec: 1.0         # 基礎間隔
  max_throttle_sec: 8.0     # 最大間隔

  # 網路設定
  retry_total: 5            # 重試次數
  http_timeout: 45.0        # 超時時間（秒）

  # PDF 大小限制
  max_pdf_bytes: 41943040   # 40 MB

  # User Agent（可修改以模擬不同瀏覽器）
  user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
```

## 執行方式

### 方法 1：使用 uv run（推薦）

```bash
# 執行爬蟲
uv run python -m src.esg_crawler

# 或簡寫
uv run -m src.esg_crawler
```

### 方法 2：透過 main.py 執行

```bash
# 啟用爬蟲功能（在 config.yaml 中設定）
# pipeline.crawl: true

uv run python main.py --stage 0
```

## CSV 檔案格式要求

S&P 500 CSV 檔案必須包含 `ticker` 欄位：

```csv
ticker,company_name
AAPL,Apple Inc.
MSFT,Microsoft Corporation
GOOGL,Alphabet Inc.
...
```

最基本只需要 `ticker` 欄位：

```csv
ticker
AAPL
MSFT
GOOGL
```

## 輸出結果

### 1. PDF 檔案

下載的 PDF 檔案會儲存在：

```
data/raw/{ticker}/{year}/{ticker}_{year}_{hash}.pdf
```

例如：
```
data/raw/AAPL/2017/AAPL_2017_a1b2c3d4.pdf
data/raw/AAPL/2018/AAPL_2018_e5f6g7h8.pdf
```

### 2. Manifest 檔案

所有下載記錄儲存在：

```
data/metadata/esg_manifest.csv
```

包含以下資訊：
- timestamp: 時間戳記
- ticker: 公司代碼
- year: 年份
- url: 原始 URL
- final_url: 最終 URL（經過重導向後）
- title: 文件標題
- mime: MIME 類型
- bytes: 檔案大小
- sha256: 檔案雜湊值
- status: 狀態（downloaded/skipped/error）
- status_detail: 詳細狀態
- saved_path: 儲存路徑

## 常見問題

### Q1: 找不到 S&P 500 CSV 檔案

**錯誤訊息：**
```
找不到 SP500 CSV 文件: ./data/sp500_2025.csv
```

**解決方法：**
1. 確認檔案存在：`ls -l data/sp500_2025.csv`
2. 如果使用其他檔案，修改 `config/config.yaml` 中的 `paths.sp500_csv`

### Q2: 沒有找到 BRAVE_API_KEY

**錯誤訊息：**
```
錯誤: 未找到 BRAVE_API_KEY 環境變數
```

**解決方法：**
在 `.env` 檔案中添加：
```bash
BRAVE_API_KEY=your_api_key_here
```

### Q3: 如何修改爬取年份？

編輯 `config/config.yaml`：

```yaml
crawler:
  start_year: 2019  # 修改為你想要的起始年份
  end_year: 2022    # 修改為你想要的結束年份
```

### Q4: 如何使用不同的公司列表？

1. 準備 CSV 檔案，確保有 `ticker` 欄位
2. 放在 `data/` 目錄下
3. 修改 `config/config.yaml`：

```yaml
paths:
  sp500_csv: "./data/your_custom_list.csv"
```

### Q5: 如何調整爬蟲速度？

修改 `config/config.yaml` 中的 throttle 設定：

```yaml
crawler:
  # 加快速度（但可能被封鎖）
  throttle_sec: 0.5

  # 或減慢速度（更安全）
  throttle_sec: 2.0
```

## 執行範例

### 範例 1：爬取 2020-2023 年的 ESG 報告

1. 修改 `config/config.yaml`：
```yaml
crawler:
  start_year: 2020
  end_year: 2023
```

2. 執行爬蟲：
```bash
uv run python -m src.esg_crawler
```

### 範例 2：使用自訂公司列表

1. 建立 `data/tech_companies.csv`：
```csv
ticker
AAPL
MSFT
GOOGL
META
AMZN
```

2. 修改 `config/config.yaml`：
```yaml
paths:
  sp500_csv: "./data/tech_companies.csv"
```

3. 執行爬蟲：
```bash
uv run python -m src.esg_crawler
```

## 監控執行進度

爬蟲會顯示即時進度（已添加實時輸出緩衝刷新）：

```
注意: 本爬蟲現在使用 config/config.yaml 進行配置
API Key 請設定在 .env 文件的 BRAVE_API_KEY 環境變數中

開始爬取 10 家公司的 ESG 報告...
總任務數: 720
==================================================

處理 1/10: AAPL
  查詢: AAPL 2013 ... 找到 15 個結果
    ✓ 下載: AAPL_2013_a1b2c3d4.pdf (2.45 MB)
  查詢: AAPL 2014 ... 找到 12 個結果
    ✓ 下載: AAPL_2014_e5f6g7h8.pdf (3.23 MB)
  完成 AAPL

處理 2/10: MSFT
  查詢: MSFT 2013 ... 找到 18 個結果
...
```

完成後會顯示統計：

```
==================================================
爬取完成!
成功下載: 125 個 PDF
跳過: 23 個
錯誤: 2 個
==================================================
Done. Manifest at: data/metadata/esg_manifest.csv
```

**注意：** 所有進度輸出都已經添加了 `flush=True`，確保在終端執行時可以即時看到進度，不會被緩衝。

## 進階設定

### 自訂搜尋關鍵詞

編輯 `config/keywords.yaml`：

```yaml
report_keywords:
  - "sustainability report"
  - "ESG report"
  - "corporate responsibility report"
  - "CSR report"
  - "annual sustainability report"  # 新增自訂關鍵詞
  - "環境報告書"                     # 可加入中文關鍵詞
```

### 公司名稱映射

如果需要使用公司全名搜尋，可以準備映射檔案：

1. 建立 `data/company_mapping.csv`：
```csv
ticker,company
AAPL,Apple Inc.
MSFT,Microsoft Corporation
```

2. 修改 `config/config.yaml`：
```yaml
crawler:
  company_map: "./data/company_mapping.csv"
```

## 測試配置

在正式執行前，可以先測試配置：

```bash
# 測試配置載入
uv run python -c "from src.config_loader import get_config; c = get_config(); print(f'S&P500 CSV: {c.sp500_csv}'); print(f'年份: {c.crawler_start_year}-{c.crawler_end_year}')"

# 檢查 CSV 檔案
uv run python -c "from src.esg_crawler import read_sp500_csv; from src.config_loader import get_config; c = get_config(); tickers = read_sp500_csv(c.sp500_csv); print(f'公司數量: {len(tickers)}'); print(f'前5家: {tickers[:5]}')"
```

## 效能建議

1. **調整並行度**：爬蟲預設單線程，確保穩定性
2. **合理設定間隔**：`throttle_sec: 1.0` 是推薦值
3. **批次處理**：可以分批爬取不同年份
4. **斷點續傳**：爬蟲會自動跳過已下載的檔案（根據 URL 和 SHA256）

## 技術支援

如有問題，請檢查：
1. 配置檔案語法是否正確（YAML 格式）
2. 環境變數是否設定
3. CSV 檔案格式是否正確
4. 網路連線是否正常

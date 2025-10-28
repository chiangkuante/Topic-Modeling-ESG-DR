# ESG 報告主題建模與數位韌性量化框架

自動化框架，用於爬取企業 ESG 報告，進行主題建模，將主題映射至數位韌性構面，並產出量化的數位韌性分數。

## 技術棧

- Python 3.13+
- UV (套件管理)
- Pandas, NumPy (數據處理)
- BERTopic (主題建模)
- OpenAI API (文本嵌入與 LLM)
- Scikit-learn, NLTK (機器學習與 NLP)

## 專案結構

```
project_root/
├── .venv/               # 虛擬環境
├── data/                # 數據目錄
│   ├── raw_reports/     # 原始下載的報告
│   ├── processed_corpus/ # 處理後的語料庫
│   ├── models/          # 儲存的模型
│   └── results/         # 最終結果
├── src/                 # 原始碼
│   ├── crawler.py       # 爬蟲與預處理模組
│   ├── topic_modeler.py # 主題建模模組
│   ├── mapper.py        # 主題映射模組
│   ├── scorer.py        # 分數計算模組
│   └── utils.py         # 輔助函數
├── notebooks/           # Jupyter Notebooks
├── tests/               # 測試程式碼
├── .env.example         # 環境變數範例
├── .gitignore           # Git 忽略檔案
├── requirements.txt     # 套件依賴
├── main.py              # 主執行腳本
└── README.md            # 專案說明
```

## 安裝步驟

### 1. 安裝 UV

Windows:
```bash
pip install uv
```

Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 建立虛擬環境

```bash
uv venv .venv
```

### 3. 啟動虛擬環境

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

### 4. 安裝依賴套件

```bash
uv pip install -r requirements.txt
```

### 5. 設定環境變數

複製 `.env.example` 為 `.env` 並填入 API 金鑰：

```bash
cp .env.example .env
```

編輯 `.env` 並加入您的 OpenAI API 金鑰：
```
OPENAI_API_KEY=your_api_key_here
```

## 使用方式

### 執行完整流程

```bash
python main.py --run-all
```

### 執行特定階段

```bash
# 只執行爬蟲
python main.py --run-crawler

# 只執行主題建模
python main.py --run-modeling

# 只執行主題映射
python main.py --run-mapping

# 只執行評分
python main.py --run-scoring
```

### 調整日誌級別

```bash
python main.py --run-all --log-level DEBUG
```

## 執行測試

```bash
pytest tests/
```

## 開發狀態

目前專案處於初始設定階段，各模組的核心功能尚在開發中。

- [x] 階段 0: 環境設定與專案結構
- [ ] 階段 1: 爬蟲與預處理模組
- [ ] 階段 2: 主題建模模組
- [ ] 階段 3: 主題映射模組
- [ ] 階段 4: 量化數位韌性分數模組
- [ ] 階段 5: 整合與主腳本
- [ ] 階段 6: 測試
- [ ] 階段 7: 文件與部署

## 授權

本專案僅供學術研究使用。

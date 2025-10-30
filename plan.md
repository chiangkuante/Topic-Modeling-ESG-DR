# To-do List: ESG 報告主題建模與數位韌性量化框架 (Python + UV)

**目標**: 建構一個自動化框架，用於爬取企業 ESG 報告，進行主題建模，將主題映射至數位韌性構面，並最終產出量化的數位韌性分數。

**技術棧**: Python, UV (套件管理),jupyter, Pandas, BERTopic, OpenAI API, Scikit-learn, NLTK。

**AI 助手規則**:
1.  **理解優先**: 在執行每個任務前，請確保完全理解其目的、輸入、輸出和相關依賴。
2.  **模組化**: 嚴格按照定義的模組邊界開發功能，確保低耦合、高內聚。
3.  **可配置性**: 將所有重要參數（如 API 金鑰、模型名稱、檔案路徑、閾值等）集中管理，使用 `.env` 檔案或 YAML 配置文件。
4.  **錯誤處理**: 為所有 I/O 操作、API 呼叫和潛在的計算錯誤添加健壯的錯誤處理邏輯 (try-except)。
5.  **日誌記錄**: 加入適當的日誌記錄 (logging 模組)，記錄關鍵步驟、錯誤和警告。
6.  **遵循語法**: 嚴格遵守 Python 語法和 PEP 8 規範。
7.  **測試驅動**: 為核心功能編寫單元測試 (Unit Tests) 和整合測試 (Integration Tests)。
8.  **版本控制**: 使用 Git 進行版本控制，遵循 `.gitignore` 檔案規則。
9.  **註解**: 使用繁體中文註解，不要用emoji，必須簡要，不繁雜

---

## 階段 0: 環境設定與專案結構

* **[✓] 任務 0.1**: 安裝 UV
    * **說明**: 根據 UV 官方文件安裝 UV 套件管理器。
    * **指令參考**: `pip install uv`
    * **狀態**: ✅ 已完成

* **[✓] 任務 0.2**: 建立虛擬環境
    * **說明**: 使用 UV 建立一個新的 Python 虛擬環境。
    * **指令參考**: `uv venv <環境名稱>` (例如: `uv venv .venv`)
    * **驗證**: 成功建立 `.venv` 目錄。
    * **狀態**: ✅ 已完成 - 使用 Python 3.12

* **[✓] 任務 0.3**: 啟動虛擬環境
    * **說明**: 啟動 UV 建立的虛擬環境。
    * **指令參考**: `source .venv/bin/activate` (Linux/macOS) 或 `.venv\Scripts\activate` (Windows)
    * **狀態**: ✅ 已完成

* **[✓] 任務 0.4**: 安裝依賴套件
    * **說明**: 使用 UV 根據 `pyproject.toml` 檔案安裝所有必要的 Python 套件。
    * **指令參考**: `uv pip install -e .`
    * **參考檔案**: `pyproject.toml`
    * **驗證**: 執行 `uv pip list` 檢查是否所有套件都已安裝。
    * **狀態**: ✅ 已完成

* **[✓] 任務 0.5**: 設定專案結構
    * **說明**: 建立標準的 Python 專案目錄結構。
    * **結構建議**:
        ```
        project_root/
        ├── .venv/               # 虛擬環境 (由 uv 創建)
        ├── data/                # 數據目錄 (受 .gitignore 保護)
        │   ├── raw/     # 原始下載的報告 (PDF)
        │   ├── metadata/        # 爬蟲 manifest
        │   ├── processed_corpus/ # 處理後的語料庫 (如 corpus.csv)
        │   ├── models/          # 儲存的模型 (BERTopic)
        │   ├── results/         # 最終結果 (分數, DRI)
        │   └── sp500.csv        # s&p500公司列表
        ├── src/                 # 原始碼目錄
        │   ├── __init__.py
        │   ├── esg_crawler.py   # ESG 報告爬蟲 (Brave API)
        │   ├── data_loader.py   # 資料載入與預處理
        │   ├── semantic_chunker.py # 語義分塊模組
        │   ├── topic_modeler.py # 主題建模模組
        │   ├── mapper.py        # 主題映射模組
        │   └── scorer.py        # 分數計算模組
        ├── notebooks/           # Jupyter Notebooks (main.ipynb:從src調用函數)
        ├── tests/               # 測試程式碼
        ├── .env                 # 環境變數 (API Keys, etc.)
        ├── .env.example         # 環境變數範例
        ├── .gitignore           # Git 忽略檔案
        ├── pyproject.toml
        ├── main.py              # 主執行腳本 (CLI 介面)
        └── plan.md              # 開發計畫、TODO list
        ```
    * **參考檔案**: `.gitignore`
    * **驗證**: 目錄結構符合建議。
    * **狀態**: ✅ 已完成

* **[✓] 任務 0.6**: 設定環境變數 (使用 python-dotenv)
    * **說明**: 使用 `.env` 檔案管理所有 API 金鑰和配置參數。
    * **步驟**:
        1. 複製 `.env.example` 為 `.env`: `cp .env.example .env`
        2. 編輯 `.env` 填入實際的 API 金鑰和配置
        3. 確認 `.env` 已被 `.gitignore` 排除（避免洩漏金鑰）
    * **必要的環境變數**:
        * `BRAVE_API_KEY` - Brave Search API 金鑰
        * `OPENAI_API_KEY` - OpenAI API 金鑰
    * **可選的環境變數**:
        * `EMBEDDING_MODEL` - 嵌入模型名稱（預設: text-embedding-3-small）
        * `LLM_MODEL` - LLM 模型名稱（預設: gpt-4o-mini）
        * 路徑設定（預設使用相對路徑）
    * **整合方式**:
        ```python
        from dotenv import load_dotenv
        import os

        # 載入環境變數
        load_dotenv()

        # 讀取環境變數
        config = {
            'brave_key': os.getenv('BRAVE_API_KEY'),
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        }
        ```
    * **狀態**: ✅ 已完成

* **[✓] 任務 0.7**: 整合 Brave 爬蟲到 main.py
    * **說明**: 將 `src/esg_crawler.py` 的功能整合到 `main.py` 作為 Stage 0 的可選功能
    * **實現方式**:
        * 新增 `stage0_crawl_pdfs()` 函數
        * 新增命令行參數: `--crawl`, `--start-year`, `--end-year`, `--max-results`, `--throttle-sec`
    * **使用方式**: `python main.py --stage 0 --crawl --start-year 2017 --end-year 2018`
    * **狀態**: ✅ 已完成 (2025-10-29)

---

## 階段 1: 資料載入與預處理模組

**資料來源**:
**方式**: Brave Search API 爬蟲 (`src/esg_crawler.py`) #已經完成，不需更改

**環境變數需求** (必須在 `.env` 中設定):
* `BRAVE_API_KEY` - 用於爬取 ESG 報告（推薦）
* `OPENAI_API_KEY` - 用於語義分塊（可選，不提供則使用簡單分塊）
* `EMBEDDING_MODEL` - 嵌入模型（預設: text-embedding-3-small）

### 子階段 1A: ESG 報告爬蟲 (`src/esg_crawler.py`) #已經完成，不需更改

* **[x] 任務 1A.1**: 實作 ESGCrawler 類別 #已經完成，不要更改，只需要讓main.ipynb可以調用
    * **說明**: 使用 Brave Search API 自動搜尋並下載企業 ESG 報告 PDF
    * **功能**:
        * 使用 Brave Search API 搜尋 ESG 報告
        * 支援批次爬取多家公司、多個年份
        * 自動過濾並下載 PDF 檔案
        * SHA-256 雜湊值避免重複下載
        * 生成詳細 manifest 記錄
        * 支援中斷續傳（resume 模式）
    * **輸出**:
        * PDF 檔案: `data/raw_reports/{ticker}/{year}/*.pdf`
        * Manifest: `data/metadata/esg_manifest.csv`
    * **狀態**: 已完成

### 子階段 1B: 資料載入與預處理 (`src/data_loader.py`)

* **[ ] 任務 1B.1**: 實現資料集探索與分析
    * **說明**: 探索資料集結構，了解檔案組織方式、公司數量、報告年份分布等。
    * **輸入**: `data/raw/` 目錄中的檔案（來自爬蟲）
    * **輸出**: 資料集統計資訊（公司數量、檔案數量、年份範圍等）
    * **重點**:
        * 分析資料集的目錄結構
        * 統計各公司的報告數量
        * 載入PDF
        * 記錄資料集基本資訊
    * **狀態**: 待實作

* **[ ] 任務 1B.2**: 實現 PDF 文本提取
    * **說明**: 從爬蟲下載的 PDF 檔案提取文本內容
    * **輸入**: PDF 檔案路徑
    * **輸出**: 提取的純文本字串
    * **工具**: PyPDF2、pdfplumber 或 pdfminer.six
    * **重點**:
        * 處理不同格式的 PDF
        * 保留文本結構
        * 錯誤處理（加密、損壞的 PDF）
    * **狀態**: 待實作

* **[ ] 任務 1B.3**: 實現文本載入與清理
    * **說明**: 載入文本資料並進行初步清理
    * **輸入**: 文本檔案路徑（TXT、CSV）或 PDF 提取文本
    * **輸出**: 清理後的純文本字串
    * **重點**:
        * 處理多餘的換行符和空格
        * 添加錯誤處理
        * 記錄載入進度和錯誤
    * **狀態**: 待實作

* **[ ] 任務 1B.4**: 實現句子過濾
    * **說明**: 過濾掉過短的句子和樣板化的法律聲明
    * **輸入**: 清理後的純文本字串
    * **輸出**: 過濾後的句子列表
    * **重點**:
        * 使用 `nltk.sent_tokenize` 分句
        * 可配置的句子最小長度閾值
        * 可配置的樣板化關鍵詞列表（如 "forward-looking statement", "safe harbor" 等）
        * 記錄過濾統計資訊
    * **狀態**: 待實作

* **[ ] 任務 1B.5**: 實現文本分塊 (Chunking)
    * **說明**: 將長文本切分成大小適中、語義連貫的文本塊，以便後續嵌入和建模
    * **輸入**: 過濾後的句子列表
    * **輸出**: 文本塊列表
    * **重點**:
        * 一定要使用`./llm_semantic_chunker.py`檔案來修改，作為方法
        * 使用 `tiktoken` 計算 token 數
        * 處理 API Rate Limit（加入 `time.sleep`）
    * **狀態**: 待實作

* **[ ] 任務 1B.6**: 建立最終語料庫
    * **說明**: 將所有處理好的文本塊，連同其元數據（公司名稱、年份、文件來源）彙集成一個 CSV 檔案
    * **輸入**: 所有公司、所有文件的文本塊列表及元數據
    * **輸出**: `data/processed_corpus/corpus.csv`（欄位: `ticker`, `year`, `text`）
    * **重點**:
        * 從檔案名稱或路徑提取公司資訊和年份
        * 確保資料完整性（無遺漏、無重複）
        * 記錄統計資訊（總文本塊數、公司數、年份分布）
    * **狀態**: 待實作

* **[ ] 任務 1B.7**: 整合資料載入器類別
    * **說明**: 將以上步驟整合為一個可執行的模組 `DataLoader` 類別
    * **主要方法**:
        * `explore_dataset()` - 探索資料集結構
        * `load_text_data()` - 載入文本資料
        * `process_files()` - 批次處理檔案（清理、過濾、分塊）
        * `build_corpus()` - 建立最終語料庫
    * **配置**: 從外部讀取 API 憑證、路徑、過濾參數等
    * **重點**: 完整的錯誤處理、日誌記錄、進度追蹤
    * **狀態**: 待實作


## 階段 2: 主題建模模組 (`src/topic_modeler.py`)

* **[ ] 任務 2.1**: 實現語料庫載入
    * **說明**: 載入 `corpus.csv` 檔案。
    * **參考**: `main.ipynb` 的 `phase2_load_corpus` 區塊。
    * **輸入**: `corpus.csv` 路徑 (可配置)。
    * **輸出**: Pandas DataFrame，包含 `text`, `ticker`, `year` 等欄位。

* **[ ] 任務 2.2**: 實現文本嵌入生成與緩存
    * **說明**: 使用指定的 OpenAI 嵌入模型生成文本嵌入向量，並實現緩存機制。
    * **參考**: `main.ipynb` 的 `phase2_embeddings` 區塊。
    * **輸入**: DataFrame 中的 `text` 列表。
    * **輸出**: NumPy 陣列 (`embeddings.npy`) 和索引檔案 (`embeddings_index.json`) 存於 `data/models/`。
    * **重點**:
        * 嵌入模型名稱 (`EMBEDDING_MODEL`) 可配置。
        * 批次大小 (`EMBEDDING_BATCH_SIZE`) 可配置。
        * 緩存路徑可配置。
        * 檢查緩存時，不僅檢查檔案是否存在，還要檢查 `embeddings_index.json` 中的模型名稱和數量是否匹配當前配置。

* **[ ] 任務 2.3**: 實現初始 BERTopic 模型訓練 (Phase 2)
    * **說明**: 使用預設參數訓練初始的 BERTopic 模型。
    * **參考**: `main.ipynb` 的 `phase2_bertopic` 區塊。
    * **輸入**: 文本列表 (`texts`), 嵌入向量 (`embeddings`)。
    * **輸出**:
        * 訓練好的 BERTopic 模型 (存於 `data/models/phase2_bertopic_model/`)。
        * 主題資訊 CSV (`data/results/phase2_topics.csv`)。
        * 帶有主題標籤的語料庫 CSV (`data/results/phase2_corpus_with_topics.csv`)。
        * (可選) 主題機率矩陣 (`data/models/phase2_doc_topic_probs.npy`)。
    * **重點**:
        * UMAP 和 HDBSCAN 的參數 (n_neighbors, min_cluster_size 等) 均設為可配置。
        * 儲存路徑可配置。
        * 計算並記錄初始指標 (離群率等)。

* **[ ] 任務 2.4**: 實現基準模型對照實驗 (可選但建議)
    * **說明**: 實作 LDA 模型作為基準，並與 Phase 2的 BERTopic 結果進行比較。
    * **參考**: `main.ipynb` 的 `Phase 2.5` 區塊。
    * **輸入**: 原始文本、嵌入向量。
    * **輸出**: 對比結果表格或 JSON 檔案 (`data/results/baseline_comparison.json`)。
    * **重點**:
        * 確保 LDA 的主題數量與 BERTopic 初始主題數可比。
        * 使用相同的指標函數 (`compute_metrics`) 進行評估。

* **[ ] 任務 2.5**: 封裝 `topic_modeler.py`
    * **說明**: 將 Phase 2的邏輯封裝成類別或函數，例如 `TopicModeler` 類別，包含 `train_initial_model()`, `optimize_topics()` 等方法。

---

## 階段 3: 主題映射模組 (`src/mapper.py`)

* **[ ] 任務 3.1**: 實現優化後模型載入
    * **說明**: 載入 Phase 2 產出的優化後 BERTopic 模型和帶有最終主題的語料庫 DataFrame。
    * **參考**: `main.ipynb` 的 `phase4_load` 區塊。
    * **輸入**: Phase 3 相關檔案路徑 (可配置)。
    * **輸出**: BERTopic 模型物件, Pandas DataFrame。

* **[ ] 任務 3.2**: 實現主題到構面映射 (含緩存)
    * **說明**: 使用 LLM 將優化後的主題映射到預定義的數位韌性構面，並實現緩存。
    * **參考**: `main.ipynb` 的 `phase4_map_topics` 區塊。
    * **輸入**: BERTopic 模型物件, 構面定義 (`DIMENSIONS`)。
    * **輸出**:
        * 主題到構面的映射字典 (`{topic_id: dimension_name}`)。
        * 緩存檔案 (`data/results/phase4_topic_dimension_map.json`)。
    * **重點**:
        * 構面定義 (`DIMENSIONS`) 可配置。
        * LLM Prompt 應包含主題關鍵詞以提高準確性。
        * LLM 模型名稱可配置。
        * 處理 LLM 回應的解析 (JSON)。
        * 緩存路徑可配置。

* **[ ] 任務 3.3**: 封裝 `mapper.py`
    * **說明**: 將映射邏輯封裝成函數或類別，例如 `TopicMapper` 類別，包含 `load_model()`, `map_topics_to_dimensions()` 等方法。

---

## 階段 4: 量化數位韌性分數模組 (`src/scorer.py`)

* **[ ] 任務 4.1**: 實現主題層級評分 (含緩存)
    * **說明**: 使用 LLM 對每個 (主題, 構面) 組合進行評分 (0-5)，採用批次和選擇性評分優化，並實現緩存。
    * **參考**: `main.ipynb` 的 `phase4_score_topics` 區塊。
    * **輸入**: BERTopic 模型物件, 主題-構面映射字典, 構面分組 (`DIMENSION_GROUPS`), 評分標準 (`SCORING_RUBRIC`)。
    * **輸出**:
        * 主題評分字典 (`{topic_id: {dimension: score}}`)。
        * 緩存檔案 (`data/results/phase4_topic_scores.json`)。
    * **重點**:
        * 構面分組 (`DIMENSION_GROUPS`) 和評分標準 (`SCORING_RUBRIC`) 可配置。
        * LLM Prompt 應包含主題關鍵詞、示例文本、要評分的構面列表及評分標準。
        * 實現批次評分邏輯 (一次 API 呼叫評估多個相關構面)。
        * 處理 LLM 回應解析 (JSON)，並進行分數驗證 (限制在 0-10)。
        * 緩存路徑可配置。
        * 使用 gpt-5-nano-2025-08-07 模型

* **[ ] 任務 4.2**: 實現文檔層級分數計算
    * **說明**: 根據文檔的主題機率分布 (如果可用) 或硬分配，加權計算每個文檔在各個數位韌性構面上的得分。
    * **參考**: `main.ipynb` 的 `phase4_calc_doc_scores` 區塊。
    * **輸入**: 帶有主題標籤的 DataFrame, 主題評分字典, (可選) 主題機率矩陣 (`probs.npy`)。
    * **輸出**: 更新後的 DataFrame，包含每個構面的分數欄位 (存為 `data/results/part4_doc_dimension_scores.csv`)。
    * **重點**:
        * 正確處理使用機率或硬分配的邏輯。
        * 輸出檔案路徑可配置。

* **[ ] 任務 4.3**: 實現數位韌性指數 (DRI) 計算
    * **說明**: 根據每個構面的分數和預定義的權重，計算最終的 DRI。聚合到實體-時間維度 (例如，公司-年份)。
    * **參考**: `main.ipynb` 的 `phase4_calc_dri` 區塊。
    * **輸入**: 包含文檔分數的 DataFrame, 構面權重 (`DIMENSION_WEIGHTS`)。
    * **輸出**: 包含 DRI 的聚合後 DataFrame (存為 `data/results/part4_entity_time_dri.csv`)。
    * **重點**:
        * 構面權重 (`DIMENSION_WEIGHTS`) 可配置。
        * 聚合的鍵 (例如 `ticker`, `year`) 應自動偵測或可配置。
        * 輸出檔案路徑可配置。

* **[ ] 任務 4.4**: 封裝 `scorer.py`
    * **說明**: 將評分和 DRI 計算邏輯封裝成函數或類別，例如 `DigitalResilienceScorer` 類別，包含 `score_topics()`, `calculate_document_scores()`, `calculate_dri()` 等方法。

---

## 階段 5: 整合與主腳本 (`main.py`)

* **[ ] 任務 5.1**: 建立主執行流程
    * **說明**: 創建 `main.py` 腳本，按順序調用 `crawler`, `topic_modeler`, `mapper`, `scorer` 模組的功能，完成從數據下載到 DRI 計算的完整流程。
    * **重點**:
        * 從配置文件或 `.env` 讀取所有配置。
        * 在模組之間傳遞必要的數據 (例如 DataFrame, 模型物件)。
        * 加入命令行參數 (使用 `argparse`) 控制執行的階段 (例如，`--run-crawler`, `--run-modeling`, `--use-cache`)。

* **[ ] 任務 5.2**: 加入日誌和進度顯示
    * **說明**: 在 `main.py` 中設定全局日誌，並在關鍵步驟使用 `tqdm` 等庫顯示進度。

---

## 階段 6: 測試 (`tests/`)

* **[ ] 任務 6.1**: 編寫單元測試
    * **說明**: 為 `utils.py` 中的輔助函數、`crawler.py` 中的文本清理/分塊函數、`scorer.py` 中的 DRI 計算邏輯等編寫單元測試。
    * **工具**: `pytest`。
    * **重點**: 測試邊界條件和預期輸出。

* **[ ] 任務 6.2**: 編寫整合測試
    * **說明**: 設計小型測試案例 (例如，使用少量文件)，測試從爬蟲到 DRI 計算的完整流程是否能順利運行。
    * **重點**: 檢查各模組之間的接口和數據傳遞是否正確。

* **[ ] 任務 6.3**: LLM Agent 輸出驗證
    * **說明**: 設計測試檢查 LLM Agent 的輸出是否符合預期的 JSON 格式，以及分數是否在 0-10 範圍內。

* **[ ] 任務 6.4**: 設定測試環境
    * **說明**: 確保測試可以在隔離的環境中運行，可能需要 mock API 呼叫或使用測試數據集。

---

## 階段 7: 文件與部署

* **[ ] 任務 7.1**: 編寫 README.md
    * **說明**: 撰寫專案說明文件，包含：
        * 專案目標
        * 安裝步驟 (使用 UV)
        * 如何運行框架 (命令行參數)
        * 配置說明
        * 模組架構
        * 輸出文件說明

* **[ ] 任務 7.2**: 封裝為套件
    * **說明**: 使用 `pyproject.toml` 將專案封裝成可安裝的 Python 套件。

* **[ ] 任務 7.3**: 建立介面
    * **說明**: 使用 `streamlit` 提供更友好的使用者介面。

# To-do List: ESG 報告主題建模與數位韌性量化框架 (Python + UV)

**目標**: 建構一個自動化框架，用於爬取企業 ESG 報告，進行主題建模，將主題映射至數位韌性構面，並最終產出量化的數位韌性分數。

**技術棧**: Python, UV (套件管理), Pandas, BERTopic, OpenAI API, Scikit-learn, NLTK。

**AI 助手規則**:
1.  **理解優先**: 在執行每個任務前，請確保完全理解其目的、輸入、輸出和相關依賴。
2.  **模組化**: 嚴格按照定義的模組邊界開發功能，確保低耦合、高內聚。
3.  **可配置性**: 將所有重要參數（如 API 金鑰、模型名稱、檔案路徑、閾值等）集中管理，建議使用 `.env` 檔案或 YAML 配置文件。
4.  **錯誤處理**: 為所有 I/O 操作、API 呼叫和潛在的計算錯誤添加健壯的錯誤處理邏輯 (try-except)。
5.  **日誌記錄**: 加入適當的日誌記錄 (logging 模組)，記錄關鍵步驟、錯誤和警告。
6.  **遵循語法**: 嚴格遵守 Python 語法和 PEP 8 規範。
7.  **測試驅動**: 為核心功能編寫單元測試 (Unit Tests) 和整合測試 (Integration Tests)。
8.  **版本控制**: 使用 Git 進行版本控制，遵循 `.gitignore` 檔案規則。
9.  **註解**: 使用繁體中文註解，不要用emoji，必須簡要，不繁雜 

---

## 階段 0: 環境設定與專案結構

* **[x] 任務 0.1**: 安裝 UV
    * **說明**: 根據 UV 官方文件安裝 UV 套件管理器。
    * **指令參考**: `curl -LsSf https://astral.sh/uv/install.sh | sh` 或 `pip install uv` (取決於系統)
    * **狀態**: 已完成 - UV 0.8.16 已安裝

* **[x] 任務 0.2**: 建立虛擬環境
    * **說明**: 使用 UV 建立一個新的 Python 虛擬環境。
    * **指令參考**: `uv venv <環境名稱>` (例如: `uv venv .venv`)
    * **驗證**: 成功建立 `.venv` 目錄。
    * **狀態**: 已完成 - 使用 Python 3.13.2

* **[x] 任務 0.3**: 啟動虛擬環境
    * **說明**: 啟動 UV 建立的虛擬環境。
    * **指令參考**: `source .venv/bin/activate` (Linux/macOS) 或 `.venv\Scripts\activate` (Windows)
    * **狀態**: 已完成 - 環境已建立，使用時需手動啟動

* **[x] 任務 0.4**: 安裝依賴套件
    * **說明**: 使用 UV 根據 `requirements.txt` 檔案安裝所有必要的 Python 套件。
    * **指令參考**: `uv pip install -r requirements.txt` 。
    * **參考檔案**: `requirements.txt`
    * **驗證**: 執行 `uv pip list` 檢查是否所有套件都已安裝。
    * **狀態**: 已建立 requirements.txt (UV 不支援 Conda 的 environment.yml 格式)

* **[x] 任務 0.5**: 設定專案結構
    * **說明**: 建立標準的 Python 專案目錄結構。
    * **結構建議**:
        ```
        project_root/
        ├── .venv/               # 虛擬環境 (由 uv 創建)
        ├── data/                # 數據目錄 (受 .gitignore 保護)
        │   ├── raw_reports/     # 原始下載的報告
        │   ├── processed_corpus/ # 處理後的語料庫 (如 corpus.csv)
        │   ├── models/          # 儲存的模型 (BERTopic)
        │   └── results/         # 最終結果 (分數, DRI)
        ├── src/                 # 原始碼目錄
        │   ├── __init__.py
        │   ├── crawler.py       # 爬蟲與預處理模組
        │   ├── topic_modeler.py # 主題建模模組
        │   ├── mapper.py        # 主題映射模組
        │   ├── scorer.py        # 分數計算模組
        │   └── utils.py         # 輔助函數
        ├── notebooks/           # Jupyter Notebooks (如 main.ipynb)
        ├── tests/               # 測試程式碼
        ├── .env                 # 環境變數 (API Keys, etc.)
        ├── .gitignore           # Git 忽略檔案
        ├── requirements.txt     # 套件依賴
        └── main.py              # 主執行腳本 (CLI 介面)
        ```
    * **參考檔案**: `.gitignore`
    * **驗證**: 目錄結構符合建議。
    * **狀態**: 已完成 - 所有目錄和核心檔案已建立

* **[x] 任務 0.6**: 設定環境變數
    * **說明**: 建立 `.env` 檔案，並將 OpenAI API 金鑰等敏感資訊放入其中。
    * **內容範例**: `OPENAI_API_KEY="sk-..."`
    * **整合**: 確保 Python 程式碼能使用 `python-dotenv` 套件讀取此檔案。
    * **狀態**: 已完成 - 已建立 .env.example 模板，使用者需複製並填入實際金鑰

---

## 階段 1: 爬蟲與預處理模組 (`src/crawler.py`)

* **[ ] 任務 1.1**: 實現公司列表獲取
    * **說明**: 獲取 Fortune Global 500 2018 年排行的前 200 家美國公司列表及其 Ticker Symbol。
    * **方法**: 編寫爬蟲從Fortune 官網存檔爬取。
    * **輸出**: Python 列表或元組，包含 200 個 Ticker Symbols。

* **[ ] 任務 1.2**: 實現ESG報告下載
    * **說明**: 根據公司列表，使用爬蟲下載 2017-01-01 至 2018-12-31 期間的ESG報告。
    * **參考**: `src/data_download.py`
    * **輸入**: 公司 Ticker 列表, 日期範圍。
    * **輸出**: 下載的 HTML 文件存放在 `data/raw_reports/<TICKER>/<FILING_TYPE>/`。
    * **重點**:
        * 加入更詳細的錯誤處理 (例如，處理超時、無文件的情況)。
        * 加入日誌記錄下載進度和錯誤。
        * 將下載路徑設為可配置。

* **[ ] 任務 1.3**: 實現 pdf 文本提取與清理
    * **說明**: 從下載的pdf文件中提取純文本，並進行初步清理。
    * **參考**: `src/data_processing.py`
    * **輸入**: pdf 文件路徑。
    * **輸出**: 清理後的純文本字串。
    * **重點**:
        * 處理多餘的換行符和空格。
        * 添加錯誤處理 (處理無法解析的 pdf)。

* **[ ] 任務 1.4**: 實現句子過濾
    * **說明**: 過濾掉過短的句子和樣板化的法律聲明。
    * **參考**: `src/data_processing.py` 的 NLTK 和關鍵詞過濾部分。
    * **輸入**: 清理後的純文本字串。
    * **輸出**: 過濾後的句子列表。
    * **重點**:
        * 使用 `nltk.sent_tokenize` 分句。
        * 可配置的句子最小長度閾值。
        * 可配置的樣板化關鍵詞列表。

* **[ ] 任務 1.5**: 實現文本分塊 (Chunking)
    * **說明**: 將長文本切分成大小適中、語義連貫的文本塊，以便後續嵌入和建模。
    * **參考**: `src/data_processing.py` 的分塊邏輯（包含結構化和語義分塊）。
    * **輸入**: 過濾後的句子列表。
    * **輸出**: 文本塊列表。
    * **重點**:
        * 使用`SemanticChunker`，參考以下網址:https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/llm_semantic_chunker.py 、 https://research.trychroma.com/evaluating-chunking
        * 使用 `tiktoken` 計算 token 數。
        * 處理 API Rate Limit (加入 `time.sleep`)。

* **[ ] 任務 1.6**: 建立最終語料庫
    * **說明**: 將所有處理好的文本塊，連同其元數據 (公司 Ticker, 年份, 文件來源) 彙集成一個 CSV 檔案。
    * **參考**: `src/data_processing.py` 的結尾部分。
    * **輸入**: 所有公司、所有文件的文本塊列表及元數據。
    * **輸出**: `data/processed_corpus/corpus.csv` (欄位: `ticker`, `year`, `doc_id`, `text`)。
    * **重點**: 確保 `year` 的提取邏輯正確 (從文件名或文件夾名)。

* **[ ] 任務 1.7**: 編寫 `crawler.py` 的主函數/類別
    * **說明**: 將以上步驟整合為一個可執行的模組，例如一個 `Crawler` 類別，包含 `download_reports()`, `process_files()`, `build_corpus()` 等方法。
    * **配置**: 從外部讀取公司列表、日期範圍、輸出路徑等。

---

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

* **[ ] 任務 6.3**: (可選) LLM Agent 輸出驗證
    * **說明**: 設計測試檢查 LLM Agent 的輸出是否符合預期的 JSON 格式，以及分數是否在 0-10 範圍內。

* **[ ] 任務 6.4**: 設定測試環境
    * **說明**: 確保測試可以在隔離的環境中運行，可能需要 mock API 呼叫或使用測試數據集。

---

## 階段 7: 文件與部署 (可選)

* **[ ] 任務 7.1**: 編寫 README.md
    * **說明**: 撰寫專案說明文件，包含：
        * 專案目標
        * 安裝步驟 (使用 UV)
        * 如何運行框架 (命令行參數)
        * 配置說明
        * 模組架構
        * 輸出文件說明

* **[ ] 任務 7.2**: (可選) 封裝為套件
    * **說明**: 使用 `pyproject.toml` 將專案封裝成可安裝的 Python 套件。

* **[ ] 任務 7.3**: (可選) 建立 API 或 CLI
    * **說明**: 使用 `FastAPI` 或 `Typer` 提供更友好的使用者介面。

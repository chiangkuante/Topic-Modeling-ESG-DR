# 快速開始指南

本指南將幫助你快速建構並運行階段0、1、2的ESG主題建模框架。

## 前置條件

1. Python 3.12+
2. OpenAI API金鑰
3. Brave Search API金鑰（如需爬取資料）
4. 已爬取的ESG報告PDF文件（位於 `data/raw/` 目錄）

## 快速安裝

### 1. 安裝UV並建立虛擬環境

```bash
# 安裝UV
pip install uv

# 建立虛擬環境
uv venv .venv

# 啟動虛擬環境（Windows）
.venv\Scripts\activate

# 啟動虛擬環境（Linux/macOS）
source .venv/bin/activate
```

### 2. 安裝依賴套件

```bash
uv pip install -e .
```

### 3. 設定環境變數

```bash
# 複製範例檔案
cp .env.example .env

# 編輯 .env 並填入你的API金鑰
# OPENAI_API_KEY=sk-xxx
# BRAVE_KEY=xxx
```

## 方法一：使用命令行（推薦）

### 執行所有階段

```bash
python main.py --stage all
```

這將依序執行：
- 階段0: 環境設定檢查
- 階段1: 資料載入與預處理
- 階段2: 主題建模

### 分階段執行

```bash
# 只執行階段0（環境檢查）
python main.py --stage 0

# 只執行階段1（資料預處理）
python main.py --stage 1

# 只執行階段2（主題建模）
python main.py --stage 2
```

### 使用語義分塊（可選）

語義分塊使用LLM來更智能地分割文本，但會增加API成本和處理時間。

```bash
python main.py --stage 1 --semantic-chunking
```

## 方法二：使用Jupyter Notebook

### 1. 安裝Jupyter

```bash
uv pip install jupyter ipykernel matplotlib
```

### 2. 啟動Notebook

```bash
jupyter notebook notebooks/main.ipynb
```

### 3. 在Notebook中執行

打開 `main.ipynb` 後，依序執行各個單元格：

1. **初始設定** - 載入環境變數和模組
2. **階段0** - 檢查環境設定
3. **階段1** - 資料載入與預處理
4. **階段2** - 主題建模
5. **結果查看** - 查看JSON輸出

## 預期輸出

### 階段1輸出

```
data/processed_corpus/corpus.csv
```

欄位包含：
- `ticker`: 公司代碼
- `year`: 年份
- `text`: 處理後的文本塊
- `source_file`: 來源文件名

### 階段2輸出

```
data/models/
  ├── phase2_bertopic_model/       # BERTopic模型文件
  ├── embeddings.npy               # 嵌入向量緩存
  └── embeddings_index.json        # 嵌入索引

data/results/
  ├── phase2_topics.csv            # 主題資訊CSV
  ├── phase2_corpus_with_topics.csv # 帶主題標籤的語料庫
  └── phase2_topic_summary.json    # 主題摘要JSON（主要輸出）
```

### JSON輸出格式

`phase2_topic_summary.json` 包含：

```json
{
  "metadata": {
    "total_topics": 25,
    "embedding_model": "text-embedding-3-small",
    "umap_params": {...},
    "hdbscan_params": {...}
  },
  "topics": [
    {
      "topic_id": 0,
      "count": 156,
      "name": "0_climate_carbon_emissions_reduction",
      "keywords": [
        {"word": "climate", "score": 0.0245},
        {"word": "carbon", "score": 0.0231}
      ]
    }
  ]
}
```

參考 `data/results/phase2_topic_summary_example.json` 查看完整範例。

## 執行時間估計

基於1000個文檔的資料集：

- **階段0**: < 1秒
- **階段1**:
  - 簡單分塊: 5-10分鐘
  - 語義分塊: 20-30分鐘（取決於API速度）
- **階段2**:
  - 生成嵌入（首次）: 10-15分鐘
  - 訓練BERTopic: 5-10分鐘
  - 使用緩存的嵌入: 5-10分鐘

## 常見問題

### Q: 如何檢查資料集是否正確？

```python
from src.data_loader import DataLoader

loader = DataLoader()
stats = loader.explore_dataset()
print(stats)
```

### Q: 嵌入生成太慢怎麼辦？

嵌入會自動緩存。首次執行後，再次運行階段2會直接使用緩存，速度會快很多。

### Q: 如何調整主題數量？

修改 HDBSCAN 參數：

```python
topic_modeler = TopicModeler(
    min_cluster_size=20,  # 增加這個值會減少主題數
    min_samples=15        # 增加這個值會讓聚類更嚴格
)
```

### Q: 出現API錯誤怎麼辦？

1. 檢查 `.env` 文件中的API金鑰是否正確
2. 檢查API額度是否充足
3. 檢查網絡連接

### Q: 如何查看日誌？

日誌會同時輸出到：
- 終端
- `esg_pipeline.log` 文件

查看日誌：

```bash
tail -f esg_pipeline.log
```

## 下一步

完成階段0、1、2後，你可以：

1. 查看 `data/results/phase2_topic_summary.json` 了解發現的主題
2. 分析主題關鍵詞，評估主題質量
3. 調整參數重新訓練模型
4. 繼續開發階段3（主題映射）和階段4（分數計算）

## 需要幫助？

查看完整文檔：[README.md](README.md)

查看開發計劃：[plan.md](plan.md)

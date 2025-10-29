#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader.py
資料載入與預處理模組
"""
import os
import logging
import pandas as pd
import PyPDF2
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import nltk
from tqdm import tqdm
import tiktoken
import time
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """資料載入與預處理類別"""

    def __init__(
        self,
        raw_data_path: str = "./data/raw",
        processed_data_path: str = "./data/processed_corpus",
        min_sentence_length: int = 50,
        boilerplate_keywords: Optional[List[str]] = None,
        max_chunk_tokens: int = 512,
        openai_api_key: Optional[str] = None
    ):
        """
        初始化資料載入器

        參數:
            raw_data_path: 原始資料路徑
            processed_data_path: 處理後資料路徑
            min_sentence_length: 句子最小長度
            boilerplate_keywords: 樣板化關鍵詞列表
            max_chunk_tokens: 最大chunk token數
            openai_api_key: OpenAI API金鑰
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.min_sentence_length = min_sentence_length
        self.max_chunk_tokens = max_chunk_tokens
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

        # 預設樣板化關鍵詞
        self.boilerplate_keywords = boilerplate_keywords or [
            "forward-looking statement",
            "safe harbor",
            "securities and exchange commission",
            "this report contains",
            "table of contents"
        ]

        # 確保nltk資源已下載
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        # 建立輸出目錄
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        logger.info("DataLoader初始化完成")

    def explore_dataset(self) -> Dict:
        """
        探索資料集結構

        返回:
            資料集統計資訊字典
        """
        logger.info("開始探索資料集...")

        stats = {
            'total_companies': 0,
            'total_files': 0,
            'companies': {},
            'year_range': set()
        }

        if not self.raw_data_path.exists():
            logger.warning(f"原始資料路徑不存在: {self.raw_data_path}")
            return stats

        # 遍歷公司目錄
        for company_dir in self.raw_data_path.iterdir():
            if not company_dir.is_dir():
                continue

            ticker = company_dir.name
            stats['companies'][ticker] = {}
            stats['total_companies'] += 1

            # 遍歷年份目錄
            for year_dir in company_dir.iterdir():
                if not year_dir.is_dir():
                    continue

                year = year_dir.name
                stats['year_range'].add(year)

                # 統計PDF文件數量
                pdf_files = list(year_dir.glob("*.pdf"))
                file_count = len(pdf_files)

                if file_count > 0:
                    stats['companies'][ticker][year] = file_count
                    stats['total_files'] += file_count

        stats['year_range'] = sorted(list(stats['year_range']))

        logger.info(f"資料集探索完成: 共 {stats['total_companies']} 家公司, "
                   f"{stats['total_files']} 個文件")
        logger.info(f"年份範圍: {stats['year_range']}")

        return stats

    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """
        從PDF文件提取文本

        參數:
            pdf_path: PDF文件路徑

        返回:
            提取的文本或None
        """
        try:
            # 先嘗試使用pdfplumber（更準確）
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                if text.strip():
                    return text

            # 如果pdfplumber失敗，嘗試PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text if text.strip() else None

        except Exception as e:
            logger.error(f"無法提取PDF文本 {pdf_path}: {e}")
            return None

    def clean_text(self, text: str) -> str:
        """
        清理文本

        參數:
            text: 原始文本

        返回:
            清理後的文本
        """
        if not text:
            return ""

        # 移除多餘的空白和換行
        text = " ".join(text.split())

        return text

    def filter_sentences(self, text: str) -> List[str]:
        """
        過濾句子

        參數:
            text: 清理後的文本

        返回:
            過濾後的句子列表
        """
        if not text:
            return []

        # 分句
        sentences = nltk.sent_tokenize(text)

        filtered_sentences = []
        for sentence in sentences:
            # 過濾太短的句子
            if len(sentence) < self.min_sentence_length:
                continue

            # 過濾樣板化內容
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in self.boilerplate_keywords):
                continue

            filtered_sentences.append(sentence)

        logger.debug(f"過濾句子: {len(sentences)} -> {len(filtered_sentences)}")

        return filtered_sentences

    def chunk_text_semantic(self, sentences: List[str]) -> List[str]:
        """
        使用語義分塊器進行文本分塊

        參數:
            sentences: 句子列表

        返回:
            文本塊列表
        """
        if not self.openai_api_key:
            logger.warning("未提供OpenAI API金鑰，使用簡單分塊")
            return self.chunk_text_simple(sentences)

        try:
            # 導入LLM語義分塊器
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from llm_semantic_chunker import LLMSemanticChunker

            # 初始化分塊器
            chunker = LLMSemanticChunker(
                organisation="openai",
                api_key=self.openai_api_key,
                model_name="gpt-4o-mini"
            )

            # 合併句子為文本
            text = " ".join(sentences)

            # 進行語義分塊
            logger.info("使用LLM進行語義分塊...")
            chunks = chunker.split_text(text)

            # 確保每個chunk不超過最大token數
            encoding = tiktoken.encoding_for_model("gpt-4")
            final_chunks = []

            for chunk in chunks:
                token_count = len(encoding.encode(chunk))
                if token_count > self.max_chunk_tokens:
                    # 如果超過，使用簡單分塊
                    sub_chunks = self._split_by_tokens(chunk, encoding)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)

            # 延遲以避免Rate Limit
            time.sleep(1)

            return final_chunks

        except Exception as e:
            logger.error(f"語義分塊失敗: {e}，使用簡單分塊")
            return self.chunk_text_simple(sentences)

    def chunk_text_simple(self, sentences: List[str]) -> List[str]:
        """
        簡單文本分塊（按token數）

        參數:
            sentences: 句子列表

        返回:
            文本塊列表
        """
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            encoding = tiktoken.get_encoding("cl100k_base")

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(encoding.encode(sentence))

            if current_tokens + sentence_tokens > self.max_chunk_tokens and current_chunk:
                # 保存當前chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # 添加最後一個chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_by_tokens(self, text: str, encoding) -> List[str]:
        """按token數分割文本"""
        tokens = encoding.encode(text)
        chunks = []

        for i in range(0, len(tokens), self.max_chunk_tokens):
            chunk_tokens = tokens[i:i + self.max_chunk_tokens]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def process_file(
        self,
        pdf_path: Path,
        ticker: str,
        year: str,
        use_semantic_chunking: bool = False
    ) -> List[Dict]:
        """
        處理單個PDF文件

        參數:
            pdf_path: PDF文件路徑
            ticker: 公司代碼
            year: 年份
            use_semantic_chunking: 是否使用語義分塊

        返回:
            處理後的文本塊列表（包含元數據）
        """
        logger.info(f"處理文件: {pdf_path}")

        # 提取文本
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            logger.warning(f"無法從 {pdf_path} 提取文本")
            return []

        # 清理文本
        cleaned_text = self.clean_text(raw_text)

        # 過濾句子
        filtered_sentences = self.filter_sentences(cleaned_text)
        if not filtered_sentences:
            logger.warning(f"{pdf_path} 沒有有效句子")
            return []

        # 文本分塊
        if use_semantic_chunking:
            chunks = self.chunk_text_semantic(filtered_sentences)
        else:
            chunks = self.chunk_text_simple(filtered_sentences)

        # 添加元數據
        processed_chunks = []
        for chunk in chunks:
            processed_chunks.append({
                'ticker': ticker,
                'year': year,
                'text': chunk,
                'source_file': pdf_path.name
            })

        logger.info(f"文件 {pdf_path.name} 處理完成: {len(chunks)} 個chunks")

        return processed_chunks

    def build_corpus(
        self,
        output_filename: str = "corpus.csv",
        use_semantic_chunking: bool = False
    ) -> pd.DataFrame:
        """
        建立最終語料庫

        參數:
            output_filename: 輸出檔名
            use_semantic_chunking: 是否使用語義分塊

        返回:
            語料庫DataFrame
        """
        logger.info("開始建立語料庫...")

        all_chunks = []
        stats = {'total_files': 0, 'total_chunks': 0, 'failed_files': 0}

        # 遍歷所有PDF文件
        for company_dir in tqdm(list(self.raw_data_path.iterdir()), desc="處理公司"):
            if not company_dir.is_dir():
                continue

            ticker = company_dir.name

            for year_dir in company_dir.iterdir():
                if not year_dir.is_dir():
                    continue

                year = year_dir.name

                for pdf_file in year_dir.glob("*.pdf"):
                    stats['total_files'] += 1

                    chunks = self.process_file(
                        pdf_file,
                        ticker,
                        year,
                        use_semantic_chunking=use_semantic_chunking
                    )

                    if chunks:
                        all_chunks.extend(chunks)
                        stats['total_chunks'] += len(chunks)
                    else:
                        stats['failed_files'] += 1

        # 建立DataFrame
        corpus_df = pd.DataFrame(all_chunks)

        # 保存到CSV
        output_path = self.processed_data_path / output_filename
        corpus_df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"語料庫建立完成!")
        logger.info(f"總文件數: {stats['total_files']}")
        logger.info(f"總chunks數: {stats['total_chunks']}")
        logger.info(f"失敗文件數: {stats['failed_files']}")
        logger.info(f"語料庫已保存至: {output_path}")

        return corpus_df


# 便捷函數
def load_corpus(corpus_path: str = "./data/processed_corpus/corpus.csv") -> pd.DataFrame:
    """
    載入已處理的語料庫

    參數:
        corpus_path: 語料庫文件路徑

    返回:
        語料庫DataFrame
    """
    logger.info(f"載入語料庫: {corpus_path}")
    df = pd.read_csv(corpus_path, encoding='utf-8')
    logger.info(f"語料庫載入完成: {len(df)} 個文本塊")
    return df

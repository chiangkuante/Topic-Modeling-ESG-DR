#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader_10k.py
10-K 報告資料載入與預處理模組
從 SEC Edgar filings 提取資料
"""
import os
import re
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
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

# 延遲載入配置，避免循環導入
_config = None

def _get_config():
    """延遲載入配置"""
    global _config
    if _config is None:
        try:
            from .config_loader import get_config
            _config = get_config()
        except Exception:
            _config = None
    return _config


class DataLoader10K:
    """10-K 報告資料載入與預處理類別"""

    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        processed_data_path: Optional[str] = None,
        min_sentence_length: Optional[int] = None,
        boilerplate_keywords: Optional[List[str]] = None,
        max_chunk_tokens: Optional[int] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        初始化10-K資料載入器

        參數:
            raw_data_path: 原始資料路徑（如果為 None，從配置讀取）
            processed_data_path: 處理後資料路徑（如果為 None，從配置讀取）
            min_sentence_length: 句子最小長度（如果為 None，從配置讀取）
            boilerplate_keywords: 樣板化關鍵詞列表（如果為 None，從配置讀取）
            max_chunk_tokens: 最大chunk token數（如果為 None，從配置讀取）
            openai_api_key: OpenAI API金鑰
        """
        # 嘗試從配置讀取預設值
        config = _get_config()

        if raw_data_path is None:
            raw_data_path = "./data/sec-edgar-filings"
        if processed_data_path is None:
            processed_data_path = config.processed_data_path if config else "./data/processed_corpus"
        if min_sentence_length is None:
            min_sentence_length = config.data_loader_min_sentence_length if config else 50
        if max_chunk_tokens is None:
            max_chunk_tokens = config.data_loader_max_chunk_tokens if config else 512
        if boilerplate_keywords is None:
            boilerplate_keywords = config.data_loader_boilerplate_keywords if config else [
                "forward-looking statement",
                "safe harbor",
                "securities and exchange commission",
                "this report contains",
                "table of contents",
                "item 1a. risk factors",
                "item 1b. unresolved staff comments",
                "page number",
                "sec file number"
            ]

        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.min_sentence_length = min_sentence_length
        self.max_chunk_tokens = max_chunk_tokens
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.boilerplate_keywords = boilerplate_keywords

        # 確保nltk資源已下載
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("下載 NLTK punkt_tab 資源...")
            nltk.download('punkt_tab', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("下載 NLTK punkt 資源...")
            nltk.download('punkt', quiet=True)

        # 建立輸出目錄
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        logger.info("DataLoader10K初始化完成")

    def explore_dataset(self) -> Dict:
        """
        探索10-K資料集結構

        返回:
            資料集統計資訊字典
        """
        logger.info("開始探索10-K資料集...")

        stats = {
            'total_companies': 0,
            'total_files': 0,
            'companies': {},
            'filing_years': set()
        }

        if not self.raw_data_path.exists():
            logger.warning(f"原始資料路徑不存在: {self.raw_data_path}")
            return stats

        # 遍歷公司目錄
        for company_dir in self.raw_data_path.iterdir():
            if not company_dir.is_dir():
                continue

            ticker = company_dir.name
            stats['companies'][ticker] = 0
            stats['total_companies'] += 1

            # 遍歷報告類型目錄（10-K）
            form_dir = company_dir / "10-K"
            if not form_dir.exists():
                continue

            # 遍歷各個提交目錄
            for filing_dir in form_dir.iterdir():
                if not filing_dir.is_dir():
                    continue

                # 檢查是否有 full-submission.txt
                submission_file = filing_dir / "full-submission.txt"
                if submission_file.exists():
                    stats['companies'][ticker] += 1
                    stats['total_files'] += 1

                    # 嘗試從檔案名稱提取年份
                    try:
                        year = self._extract_year_from_filing(submission_file)
                        if year:
                            stats['filing_years'].add(year)
                    except:
                        pass

        stats['filing_years'] = sorted(list(stats['filing_years']))

        logger.info(f"資料集探索完成: 共 {stats['total_companies']} 家公司, "
                   f"{stats['total_files']} 個10-K文件")
        if stats['filing_years']:
            logger.info(f"提交年份範圍: {stats['filing_years']}")

        return stats

    def _extract_year_from_filing(self, file_path: Path) -> Optional[str]:
        """從10-K提交文件中提取年份"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 讀取前1000個字符來查找年份資訊
                content = f.read(1000)

                # 查找 "CONFORMED PERIOD OF REPORT:" 行
                match = re.search(r'CONFORMED PERIOD OF REPORT:\s*(\d{4})\d{4}', content)
                if match:
                    return match.group(1)

        except Exception as e:
            logger.debug(f"無法從 {file_path} 提取年份: {e}")

        return None

    def extract_text_from_10k(self, file_path: Path) -> Optional[str]:
        """
        從10-K提交文件提取文本

        參數:
            file_path: 10-K文件路徑

        返回:
            提取的文本或None
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 分離 header 和主要內容
            # 查找 </SEC-HEADER> 標記
            header_end = content.find('</SEC-HEADER>')
            if header_end != -1:
                content = content[header_end + len('</SEC-HEADER>'):]

            # 移除 XML/XBRL 標記
            # 使用 BeautifulSoup 解析 HTML/XML
            # 嘗試不同的解析器
            try:
                soup = BeautifulSoup(content, 'html.parser')
            except Exception:
                try:
                    soup = BeautifulSoup(content, 'lxml')
                except Exception:
                    # 如果解析失敗，使用簡單的文本提取
                    logger.warning(f"無法使用 BeautifulSoup 解析 {file_path}，使用簡單文本提取")
                    # 移除 HTML 標記
                    import re
                    text = re.sub(r'<[^>]+>', ' ', content)
                    return text

            # 移除 script 和 style 元素
            for script in soup(["script", "style"]):
                script.decompose()

            # 取得文本
            text = soup.get_text()

            # 清理多餘的空白字符
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return text if text else None

        except Exception as e:
            logger.error(f"無法提取10-K文本 {file_path}: {e}")
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

        # 移除特殊字符和數字串（保留基本標點）
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?\-\(\)]+', ' ', text)

        # 移除過長的數字序列
        text = re.sub(r'\d{10,}', '', text)

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

            # 過濾包含過多數字的句子（可能是表格數據）
            digit_ratio = sum(c.isdigit() for c in sentence) / len(sentence)
            if digit_ratio > 0.5:
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
        file_path: Path,
        ticker: str,
        use_semantic_chunking: bool = False
    ) -> List[Dict]:
        """
        處理單個10-K文件

        參數:
            file_path: 10-K文件路徑
            ticker: 公司代碼
            use_semantic_chunking: 是否使用語義分塊

        返回:
            處理後的文本塊列表（包含元數據）
        """
        logger.info(f"處理文件: {file_path}")

        # 提取文本
        raw_text = self.extract_text_from_10k(file_path)
        if not raw_text:
            logger.warning(f"無法從 {file_path} 提取文本")
            return []

        # 清理文本
        cleaned_text = self.clean_text(raw_text)

        # 過濾句子
        filtered_sentences = self.filter_sentences(cleaned_text)
        if not filtered_sentences:
            logger.warning(f"{file_path} 沒有有效句子")
            return []

        # 文本分塊
        if use_semantic_chunking:
            chunks = self.chunk_text_semantic(filtered_sentences)
        else:
            chunks = self.chunk_text_simple(filtered_sentences)

        # 提取年份
        year = self._extract_year_from_filing(file_path)

        # 添加元數據
        processed_chunks = []
        for chunk in chunks:
            processed_chunks.append({
                'ticker': ticker,
                'year': year or 'unknown',
                'text': chunk,
                'source_file': file_path.name,
                'filing_type': '10-K'
            })

        logger.info(f"文件 {file_path.name} 處理完成: {len(chunks)} 個chunks")

        return processed_chunks

    def build_corpus(
        self,
        output_filename: str = "corpus_10k.csv",
        use_semantic_chunking: bool = False,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        建立10-K語料庫

        參數:
            output_filename: 輸出檔名
            use_semantic_chunking: 是否使用語義分塊
            tickers: 要處理的公司代碼列表（如果為 None，處理全部）

        返回:
            語料庫DataFrame
        """
        logger.info("開始建立10-K語料庫...")

        all_chunks = []
        stats = {'total_files': 0, 'total_chunks': 0, 'failed_files': 0}

        # 遍歷公司目錄
        company_dirs = list(self.raw_data_path.iterdir())

        # 如果指定了特定公司，只處理這些公司
        if tickers:
            company_dirs = [d for d in company_dirs if d.name in tickers]

        for company_dir in tqdm(company_dirs, desc="處理公司"):
            if not company_dir.is_dir():
                continue

            ticker = company_dir.name

            # 10-K目錄
            form_dir = company_dir / "10-K"
            if not form_dir.exists():
                continue

            # 遍歷各個提交
            for filing_dir in form_dir.iterdir():
                if not filing_dir.is_dir():
                    continue

                submission_file = filing_dir / "full-submission.txt"
                if not submission_file.exists():
                    continue

                stats['total_files'] += 1

                chunks = self.process_file(
                    submission_file,
                    ticker,
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

        logger.info(f"10-K語料庫建立完成!")
        logger.info(f"總文件數: {stats['total_files']}")
        logger.info(f"總chunks數: {stats['total_chunks']}")
        logger.info(f"失敗文件數: {stats['failed_files']}")
        logger.info(f"語料庫已保存至: {output_path}")

        return corpus_df


# 便捷函數
def load_corpus_10k(corpus_path: str = "./data/processed_corpus/corpus_10k.csv") -> pd.DataFrame:
    """
    載入已處理的10-K語料庫

    參數:
        corpus_path: 語料庫文件路徑

    返回:
        語料庫DataFrame
    """
    logger.info(f"載入10-K語料庫: {corpus_path}")
    df = pd.read_csv(corpus_path, encoding='utf-8')
    logger.info(f"語料庫載入完成: {len(df)} 個文本塊")
    return df

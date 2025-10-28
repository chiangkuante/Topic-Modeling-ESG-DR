# 爬蟲與預處理模組
# 負責下載 ESG 報告、提取文本、清理和分塊處理

import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Crawler:
    """ESG 報告爬蟲與預處理類別"""

    def __init__(self, config: Dict):
        """
        初始化爬蟲

        Args:
            config: 配置字典，包含 API 金鑰、路徑等參數
        """
        self.config = config
        logger.info("Crawler 初始化完成")

    def download_reports(self, tickers: List[str], start_date: str, end_date: str) -> None:
        """
        下載 ESG 報告

        Args:
            tickers: 公司代碼列表
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
        """
        logger.info(f"開始下載 {len(tickers)} 家公司的報告")
        # TODO: 實現下載邏輯
        pass

    def process_files(self) -> None:
        """處理下載的文件，提取文本並清理"""
        logger.info("開始處理文件")
        # TODO: 實現文本提取與清理邏輯
        pass

    def build_corpus(self) -> Path:
        """
        建立最終語料庫

        Returns:
            語料庫 CSV 檔案路徑
        """
        logger.info("開始建立語料庫")
        # TODO: 實現語料庫建立邏輯
        pass

# 輔助函數模組

import logging
from pathlib import Path
from typing import Any, Dict
import json

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """
    載入配置檔案

    Args:
        config_path: 配置檔案路徑

    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功載入配置: {config_path}")
        return config
    except Exception as e:
        logger.error(f"載入配置失敗: {e}")
        raise


def save_json(data: Any, output_path: Path) -> None:
    """
    儲存資料為 JSON 檔案

    Args:
        data: 要儲存的資料
        output_path: 輸出檔案路徑
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"成功儲存至: {output_path}")
    except Exception as e:
        logger.error(f"儲存失敗: {e}")
        raise


def load_json(input_path: Path) -> Any:
    """
    載入 JSON 檔案

    Args:
        input_path: 輸入檔案路徑

    Returns:
        載入的資料
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功載入: {input_path}")
        return data
    except Exception as e:
        logger.error(f"載入失敗: {e}")
        raise


def setup_logging(log_level: str = "INFO") -> None:
    """
    設定日誌系統

    Args:
        log_level: 日誌級別
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

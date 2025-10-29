#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_loader.py
資料載入模組測試
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

# 添加專案根目錄到路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import DataLoader


@pytest.fixture
def data_loader():
    """建立DataLoader實例"""
    return DataLoader(
        raw_data_path="./data/raw",
        processed_data_path="./data/processed_corpus",
        min_sentence_length=50,
        max_chunk_tokens=512
    )


def test_data_loader_initialization(data_loader):
    """測試DataLoader初始化"""
    assert data_loader is not None
    assert data_loader.min_sentence_length == 50
    assert data_loader.max_chunk_tokens == 512


def test_clean_text(data_loader):
    """測試文本清理"""
    text = "This is a    test   text with   extra    spaces."
    cleaned = data_loader.clean_text(text)
    assert "    " not in cleaned
    assert cleaned == "This is a test text with extra spaces."


def test_filter_sentences_min_length(data_loader):
    """測試句子過濾（最小長度）"""
    text = "Short. This is a much longer sentence that should pass the minimum length requirement."
    sentences = data_loader.filter_sentences(text)
    # "Short." 應該被過濾掉
    assert len(sentences) > 0
    assert all(len(s) >= data_loader.min_sentence_length for s in sentences)


def test_filter_sentences_boilerplate(data_loader):
    """測試句子過濾（樣板化內容）"""
    text = "This report contains forward-looking statements. This is a normal sentence about business operations."
    sentences = data_loader.filter_sentences(text)
    # 包含 "forward-looking statement" 的句子應該被過濾掉
    assert not any("forward-looking" in s.lower() for s in sentences)


def test_chunk_text_simple(data_loader):
    """測試簡單文本分塊"""
    sentences = [
        "This is the first sentence." * 20,
        "This is the second sentence." * 20,
        "This is the third sentence." * 20
    ]
    chunks = data_loader.chunk_text_simple(sentences)
    assert len(chunks) > 0
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_explore_dataset(data_loader):
    """測試資料集探索"""
    stats = data_loader.explore_dataset()
    assert isinstance(stats, dict)
    assert 'total_companies' in stats
    assert 'total_files' in stats
    assert 'companies' in stats
    assert 'year_range' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

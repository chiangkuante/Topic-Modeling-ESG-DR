# 測試 utils 模組

import pytest
from pathlib import Path
import json
import tempfile
from src.utils import save_json, load_json


def test_save_and_load_json():
    """測試 JSON 儲存和載入功能"""
    test_data = {
        'name': 'test',
        'value': 123,
        'items': ['a', 'b', 'c']
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / 'test.json'

        # 測試儲存
        save_json(test_data, test_path)
        assert test_path.exists()

        # 測試載入
        loaded_data = load_json(test_path)
        assert loaded_data == test_data


def test_load_json_file_not_found():
    """測試載入不存在的檔案"""
    with pytest.raises(Exception):
        load_json(Path('nonexistent.json'))

"""
資料載入服務測試模組
Test module for Data Loader Service

根據系統設計文檔的測試覆蓋率要求 (>= 90%)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from typing import Iterator, List, Dict
import json

from datasets import Dataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.services.data_loader import (
    DataLoadError,
    DataLoaderInterface,
    OpenCoderDataLoader,
    DataLoaderFactory
)
from src.models.dataset import (
    OriginalRecord,
    DatasetMetadata,
    Language,
    ComplexityLevel
)


class TestDataLoadError:
    """測試資料載入錯誤類別"""
    
    def test_data_load_error_creation(self) -> None:
        """測試 DataLoadError 建立"""
        error = DataLoadError("測試錯誤訊息")
        assert str(error) == "測試錯誤訊息"
        assert isinstance(error, Exception)


class TestOpenCoderDataLoader:
    """測試 OpenCoder 資料載入器"""
    
    @pytest.fixture
    def mock_dataset_record(self) -> Dict[str, object]:
        """模擬資料集記錄"""
        return {
            "instruction": "請解釋這段 Python 程式碼",
            "output": "這段程式碼實作了一個簡單的排序算法",
            "tag": "basic_programming",
            "flags": {
                "refusal": False,
                "unsolicited": False,
                "nsfw": False,
                "pii": False,
                "disclaimer": False
            }
        }
    
    @pytest.fixture
    def mock_dataset_info(self) -> Dict[str, object]:
        """模擬資料集資訊"""
        return {
            "config_name": "educational_instruct",
            "description": "Educational instruction dataset",
            "splits": {
                "train": {
                    "num_examples": 1000
                }
            },
            "version": {
                "version_str": "1.0.0"
            }
        }
    
    @pytest.fixture
    def data_loader(self) -> OpenCoderDataLoader:
        """建立測試用的資料載入器"""
        return OpenCoderDataLoader()
    
    def test_initialization(self, data_loader: OpenCoderDataLoader) -> None:
        """測試初始化"""
        assert hasattr(data_loader, 'logger')
        assert data_loader.logger.name.endswith('OpenCoderDataLoader')
    
    def test_parse_record_success(self, data_loader: OpenCoderDataLoader, mock_dataset_record: Dict[str, object]) -> None:
        """測試記錄解析成功"""
        result = data_loader._parse_record(mock_dataset_record, 0, "test_dataset")
        
        assert isinstance(result, OriginalRecord)
        assert result.id == "test_dataset_0"
        assert result.question == "請解釋這段 Python 程式碼"
        assert result.answer == "這段程式碼實作了一個簡單的排序算法"
        assert result.source_dataset == "test_dataset"
        assert result.complexity_level == ComplexityLevel.MEDIUM
        assert result.metadata["tag"] == "basic_programming"
        assert result.metadata["source_index"] == 0
    
    def test_parse_record_missing_instruction(self, data_loader: OpenCoderDataLoader) -> None:
        """測試缺少 instruction 的記錄"""
        record = {
            "output": "回答內容",
            "tag": "test"
        }
        
        with pytest.raises(ValueError, match="Record 0 missing instruction or output"):
            data_loader._parse_record(record, 0, "test_dataset")
    
    def test_parse_record_missing_output(self, data_loader: OpenCoderDataLoader) -> None:
        """測試缺少 output 的記錄"""
        record = {
            "instruction": "問題內容",
            "tag": "test"
        }
        
        with pytest.raises(ValueError, match="Record 0 missing instruction or output"):
            data_loader._parse_record(record, 0, "test_dataset")
    
    def test_parse_record_empty_instruction(self, data_loader: OpenCoderDataLoader) -> None:
        """測試空 instruction"""
        record = {
            "instruction": "",
            "output": "回答內容",
            "tag": "test"
        }
        
        with pytest.raises(ValueError, match="Record 0 missing instruction or output"):
            data_loader._parse_record(record, 0, "test_dataset")
    
    def test_parse_record_without_flags(self, data_loader: OpenCoderDataLoader) -> None:
        """測試沒有 flags 的記錄"""
        record = {
            "instruction": "問題內容",
            "output": "回答內容",
            "tag": "test"
        }
        
        result = data_loader._parse_record(record, 0, "test_dataset")
        
        assert result.metadata["flags"] == {}
    
    def test_infer_complexity_simple(self, data_loader: OpenCoderDataLoader) -> None:
        """測試推測簡單複雜度"""
        test_cases = ["simple_task", "basic_function", "easy_problem"]
        
        for tag in test_cases:
            complexity = data_loader._infer_complexity(tag)
            assert complexity == ComplexityLevel.SIMPLE
    
    def test_infer_complexity_complex(self, data_loader: OpenCoderDataLoader) -> None:
        """測試推測複雜複雜度"""
        test_cases = ["complex_algorithm", "advanced_features", "hard_problem"]
        
        for tag in test_cases:
            complexity = data_loader._infer_complexity(tag)
            assert complexity == ComplexityLevel.COMPLEX
    
    def test_infer_complexity_medium_default(self, data_loader: OpenCoderDataLoader) -> None:
        """測試預設中等複雜度"""
        test_cases = ["normal_task", "regular_function", "standard_problem"]
        
        for tag in test_cases:
            complexity = data_loader._infer_complexity(tag)
            assert complexity == ComplexityLevel.MEDIUM
    
    @patch('src.services.data_loader.pa')
    def test_load_dataset_success(self, mock_pa: Mock, data_loader: OpenCoderDataLoader, mock_dataset_record: Dict[str, object]) -> None:
        """測試資料集載入成功"""
        # 模擬 PyArrow 
        mock_table = Mock()
        mock_table.to_pylist.return_value = [mock_dataset_record]
        
        mock_reader = Mock()
        mock_reader.read_all.return_value = mock_table
        
        mock_pa.ipc.RecordBatchFileReader.return_value = mock_reader
        mock_pa.memory_map.return_value.__enter__.return_value = Mock()
        
        # 模擬找到檔案
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = [Path("test.arrow")]
            
            result = list(data_loader.load_dataset("test_path"))
            
            assert len(result) == 1
            assert isinstance(result[0], OriginalRecord)
    
    @patch('src.services.data_loader.pa')
    @patch('pathlib.Path.glob')
    def test_load_dataset_with_train_split(self, mock_glob: Mock, mock_pa: Mock, data_loader: OpenCoderDataLoader, mock_dataset_record: Dict[str, object]) -> None:
        """測試載入資料集"""
        mock_glob.return_value = [Path("test.arrow")]
        
        mock_table = Mock()
        mock_table.to_pylist.return_value = [mock_dataset_record]
        
        mock_reader = Mock()
        mock_reader.read_all.return_value = mock_table
        
        mock_pa.ipc.RecordBatchFileReader.return_value = mock_reader
        mock_pa.memory_map.return_value.__enter__.return_value = Mock()
        
        result = list(data_loader.load_dataset("test_path"))
        
        assert len(result) == 1
        assert isinstance(result[0], OriginalRecord)
    
    @patch('src.services.data_loader.load_from_disk')
    def test_load_dataset_parse_error_handling(self, mock_load_from_disk: Mock, data_loader: OpenCoderDataLoader) -> None:
        """測試記錄解析錯誤處理"""
        # 建立有問題的記錄（缺少必要欄位）
        bad_record = {"instruction": "問題"}  # 缺少 output
        good_record = {
            "instruction": "好的問題",
            "output": "好的回答",
            "tag": "good"
        }
        
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.__iter__ = Mock(return_value=iter([bad_record, good_record]))
        mock_load_from_disk.return_value = mock_dataset
        
        with patch.object(data_loader, 'logger') as mock_logger:
            result = list(data_loader.load_dataset("test_path"))
            
            # 應該只有一個有效記錄
            assert len(result) == 1
            assert result[0].question == "好的問題"
            
            # 應該記錄警告
            mock_logger.warning.assert_called_once()
    
    @patch('src.services.data_loader.load_from_disk')
    def test_load_dataset_load_error(self, mock_load_from_disk: Mock, data_loader: OpenCoderDataLoader) -> None:
        """測試資料集載入錯誤"""
        mock_load_from_disk.side_effect = Exception("檔案不存在")
        
        with pytest.raises(DataLoadError, match="Failed to load dataset from test_path"):
            list(data_loader.load_dataset("test_path"))
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_get_metadata_with_info_file(self, mock_exists: Mock, mock_file: Mock, data_loader: OpenCoderDataLoader, mock_dataset_info: Dict[str, object]) -> None:
        """測試從 info 檔案取得元資料"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(mock_dataset_info)
        
        metadata = data_loader.get_metadata("test_path")
        
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.name == "educational_instruct"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Educational instruction dataset"
        assert metadata.total_records == 1000
        assert metadata.processed_records == 0
        assert metadata.source_language == Language.ENGLISH
        assert metadata.target_language == Language.TRADITIONAL_CHINESE
    
    @patch('pathlib.Path.exists')
    def test_get_metadata_without_info_file(self, mock_exists: Mock, data_loader: OpenCoderDataLoader) -> None:
        """測試沒有 info 檔案時取得基本元資料"""
        mock_exists.return_value = False
        
        metadata = data_loader.get_metadata("test_dataset_name")
        
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.name == "test_dataset_name"
        assert metadata.version == "unknown"
        assert metadata.description == "OpenCoder dataset"
        assert metadata.total_records == 0
        assert metadata.processed_records == 0
        assert metadata.source_language == Language.ENGLISH
        assert metadata.target_language == Language.TRADITIONAL_CHINESE
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_get_metadata_json_parse_error(self, mock_exists: Mock, mock_file: Mock, data_loader: OpenCoderDataLoader) -> None:
        """測試 JSON 解析錯誤"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = "無效的 JSON"
        
        with pytest.raises(DataLoadError, match="Failed to get metadata"):
            data_loader.get_metadata("test_path")


class TestDataLoaderFactory:
    """測試資料載入器工廠"""
    
    def test_create_opencoder_loader(self) -> None:
        """測試建立 OpenCoder 載入器"""
        loader = DataLoaderFactory.create_loader("opencoder")
        assert isinstance(loader, OpenCoderDataLoader)
    
    def test_create_opencoder_loader_case_insensitive(self) -> None:
        """測試大小寫不敏感的載入器建立"""
        loader = DataLoaderFactory.create_loader("OPENCODER")
        assert isinstance(loader, OpenCoderDataLoader)
        
        loader = DataLoaderFactory.create_loader("OpenCoder")
        assert isinstance(loader, OpenCoderDataLoader)
    
    def test_create_unsupported_loader(self) -> None:
        """測試建立不支援的載入器"""
        with pytest.raises(ValueError, match="Unsupported dataset type: unknown"):
            DataLoaderFactory.create_loader("unknown")


class TestDataLoaderIntegration:
    """資料載入器整合測試"""
    
    @pytest.mark.integration
    @patch('src.services.data_loader.load_from_disk')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_end_to_end_workflow(self, mock_exists: Mock, mock_file: Mock, mock_load_from_disk: Mock) -> None:
        """測試端到端工作流程"""
        # 設置模擬資料
        dataset_records = [
            {
                "instruction": "如何在 Python 中建立清單？",
                "output": "使用方括號建立清單：my_list = [1, 2, 3]",
                "tag": "basic_python",
                "flags": {"refusal": False, "nsfw": False}
            },
            {
                "instruction": "解釋 Python 裝飾器",
                "output": "裝飾器是修改函數行為的語法糖",
                "tag": "advanced_python", 
                "flags": {"refusal": False, "nsfw": False}
            }
        ]
        
        dataset_info = {
            "config_name": "test_dataset",
            "description": "測試資料集",
            "splits": {"train": {"num_examples": 2}},
            "version": {"version_str": "1.0.0"}
        }
        
        # 設置模擬
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.__iter__ = Mock(return_value=iter(dataset_records))
        mock_load_from_disk.return_value = mock_dataset
        
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(dataset_info)
        
        # 建立載入器並測試
        loader = DataLoaderFactory.create_loader("opencoder")
        
        # 測試元資料
        metadata = loader.get_metadata("test_path")
        assert metadata.name == "test_dataset"
        assert metadata.total_records == 2
        
        # 測試資料載入
        records = list(loader.load_dataset("test_path"))
        assert len(records) == 2
        
        # 驗證第一筆記錄
        record1 = records[0]
        assert record1.question == "如何在 Python 中建立清單？"
        assert record1.answer == "使用方括號建立清單：my_list = [1, 2, 3]"
        assert record1.complexity_level == ComplexityLevel.MEDIUM
        
        # 驗證第二筆記錄  
        record2 = records[1]
        assert record2.question == "解釋 Python 裝飾器"
        assert record2.answer == "裝飾器是修改函數行為的語法糖"
        assert record2.complexity_level == ComplexityLevel.COMPLEX
    
    @pytest.mark.integration
    def test_abstract_interface_compliance(self) -> None:
        """測試抽象介面符合性"""
        loader = OpenCoderDataLoader()
        
        # 驗證實作了所有抽象方法
        assert hasattr(loader, 'load_dataset')
        assert hasattr(loader, 'get_metadata')
        assert callable(loader.load_dataset)
        assert callable(loader.get_metadata)
        
        # 驗證是 DataLoaderInterface 的實例
        assert isinstance(loader, DataLoaderInterface)


# 效能測試
class TestDataLoaderPerformance:
    """資料載入器效能測試"""
    
    @pytest.mark.performance
    @patch('src.services.data_loader.load_from_disk')
    def test_large_dataset_loading_performance(self, mock_load_from_disk: Mock, benchmark) -> None:
        """測試大型資料集載入效能"""
        # 建立大量模擬記錄
        large_dataset = [
            {
                "instruction": f"問題 {i}",
                "output": f"回答 {i}",
                "tag": "performance_test",
                "flags": {}
            }
            for i in range(10000)
        ]
        
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.__iter__ = Mock(return_value=iter(large_dataset))
        mock_load_from_disk.return_value = mock_dataset
        
        loader = OpenCoderDataLoader()
        
        # 效能測試
        def load_all():
            return list(loader.load_dataset("test_path"))
        
        result = benchmark(load_all)
        assert len(result) == 10000
    
    @pytest.mark.performance
    def test_complexity_inference_performance(self, benchmark) -> None:
        """測試複雜度推測效能"""
        loader = OpenCoderDataLoader()
        
        test_tags = [
            "simple_task", "basic_function", "easy_problem",
            "complex_algorithm", "advanced_features", "hard_problem",
            "normal_task", "regular_function", "standard_problem"
        ] * 1000  # 9000 個標籤
        
        def infer_all():
            return [loader._infer_complexity(tag) for tag in test_tags]
        
        result = benchmark(infer_all)
        assert len(result) == 9000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

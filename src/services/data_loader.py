"""
資料載入服務模組
Data Loading Service Module

負責從各種來源載入和處理資料集
"""

import json
import logging
from pathlib import Path
from typing import List, Iterator, Optional, Dict, Union, cast
from abc import ABC, abstractmethod

from ..models.dataset import (
    OriginalRecord, 
    DatasetMetadata, 
    Language,
    ComplexityLevel
)


logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """資料載入錯誤"""
    pass


class DataLoaderInterface(ABC):
    """資料載入器介面"""
    
    @abstractmethod
    def load_dataset(self, source_path: str) -> Iterator[OriginalRecord]:
        """載入資料集"""
        pass
    
    @abstractmethod
    def get_metadata(self, source_path: str) -> DatasetMetadata:
        """取得資料集元資料"""
        pass


class OpenCoderDataLoader(DataLoaderInterface):
    """OpenCoder 資料集載入器"""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__ + ".OpenCoderDataLoader")
    
    def load_dataset(self, source_path: str) -> Iterator[OriginalRecord]:
        """載入 OpenCoder 資料集"""
        try:
            self.logger.info(f"Loading dataset from {source_path}")
            
            # 使用 pyarrow 直接讀取 Arrow 檔案
            import pyarrow as pa
            
            # 尋找 Arrow 檔案
            arrow_files = list(Path(source_path).glob("**/*.arrow"))
            if not arrow_files:
                raise DataLoadError(f"No arrow files found in {source_path}")
            
            arrow_file = arrow_files[0]  # 使用第一個找到的檔案
            
            # 讀取 Arrow 檔案
            with pa.memory_map(str(arrow_file), 'r') as source:
                reader = pa.ipc.RecordBatchFileReader(source)
                table = reader.read_all()
            
            # 轉換為 Python 記錄
            records = table.to_pylist()
            
            # 遍歷並解析記錄
            for idx, record_dict in enumerate(records):
                try:
                    # 型別轉換
                    record = cast(Dict[str, object], record_dict)
                    original_record = self._parse_record(record, idx, source_path)
                    yield original_record
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse record {idx}: {e}")
                    continue
                    
        except Exception as e:
            raise DataLoadError(f"Failed to load dataset from {source_path}: {e}")
    
    def _parse_record(self, record: Dict[str, object], idx: int, source_path: str) -> OriginalRecord:
        """解析單一記錄"""
        # 從記錄中提取基本資料
        instruction = str(record.get("instruction", ""))
        output = str(record.get("output", ""))
        tag = str(record.get("tag", ""))
        
        if not instruction or not output:
            raise ValueError(f"Record {idx} missing instruction or output")
        
        # 處理 flags - 確保型別安全
        flags_raw = record.get("flags", {})
        flags: Dict[str, Union[str, int, float, bool]] = {}
        if isinstance(flags_raw, dict):
            for key, value in flags_raw.items():
                if isinstance(value, (str, int, float, bool)):
                    flags[str(key)] = value
        
        # 建立元資料
        metadata: Dict[str, Union[str, int, float, bool]] = {
            "tag": tag,
            "source_index": idx,
        }
        metadata.update(flags)
        
        # 根據 tag 推測複雜度
        complexity = self._infer_complexity(tag)
        
        # 從路徑提取資料集名稱
        dataset_name = Path(source_path).name
        
        return OriginalRecord(
            id=f"{dataset_name}_{idx}",
            question=instruction,
            answer=output,
            source_dataset=dataset_name,
            metadata=metadata,
            complexity_level=complexity
        )
    
    def _infer_complexity(self, tag: str) -> ComplexityLevel:
        """根據標籤推測複雜度"""
        tag_lower = tag.lower()
        
        if any(keyword in tag_lower for keyword in ["simple", "basic", "easy"]):
            return ComplexityLevel.SIMPLE
        elif any(keyword in tag_lower for keyword in ["complex", "advanced", "hard"]):
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.MEDIUM
    
    def get_metadata(self, source_path: str) -> DatasetMetadata:
        """取得資料集元資料"""
        try:
            # 讀取 dataset_info.json
            info_path = Path(source_path) / "train" / "dataset_info.json"
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
                
                # 型別安全的資料提取
                config_name = info_data.get("config_name")
                dataset_name = str(config_name) if config_name is not None else Path(source_path).name
                
                version_info = info_data.get("version", {})
                version = str(version_info.get("version_str", "unknown")) if isinstance(version_info, dict) else "unknown"
                
                description = str(info_data.get("description", ""))
                
                splits_info = info_data.get("splits", {})
                total_records = 0
                if isinstance(splits_info, dict):
                    train_info = splits_info.get("train", {})
                    if isinstance(train_info, dict):
                        num_examples = train_info.get("num_examples", 0)
                        total_records = int(num_examples) if isinstance(num_examples, (int, float)) else 0
                
                return DatasetMetadata(
                    name=dataset_name,
                    version=version,
                    description=description,
                    total_records=total_records,
                    processed_records=0,
                    source_language=Language.ENGLISH,
                    target_language=Language.TRADITIONAL_CHINESE
                )
            else:
                # 如果沒有 info 文件，創建基本元資料
                dataset_name = Path(source_path).name
                return DatasetMetadata(
                    name=dataset_name,
                    version="unknown",
                    description="OpenCoder dataset",
                    total_records=0,
                    processed_records=0,
                    source_language=Language.ENGLISH,
                    target_language=Language.TRADITIONAL_CHINESE
                )
                
        except Exception as e:
            raise DataLoadError(f"Failed to get metadata for {source_path}: {e}")


class DataLoaderFactory:
    """資料載入器工廠"""
    
    @staticmethod
    def create_loader(dataset_type: str) -> DataLoaderInterface:
        """建立資料載入器"""
        if dataset_type.lower() == "opencoder":
            return OpenCoderDataLoader()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")


__all__ = [
    "DataLoadError",
    "DataLoaderInterface", 
    "OpenCoderDataLoader",
    "DataLoaderFactory"
]

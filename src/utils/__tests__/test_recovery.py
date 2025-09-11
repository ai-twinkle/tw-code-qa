"""
簡潔恢復管理器測試
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from src.utils.recovery import RecoveryManager
from src.models.dataset import ProcessedRecord, ProcessingStatus, ComplexityLevel
from src.models.dataset import OriginalRecord, TranslationResult, QAExecutionResult, Language


class TestRecoveryManager:
    """測試恢復管理器"""
    
    @pytest.fixture
    def temp_dir(self):
        """創建臨時目錄"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def recovery_manager(self, temp_dir):
        """創建恢復管理器實例"""
        return RecoveryManager(temp_dir)
    
    @pytest.fixture
    def sample_record(self):
        """創建樣本記錄"""
        original_record = OriginalRecord(
            id="test_001",
            question="What is Python?",
            answer="Python is a programming language",
            source_dataset="test_dataset",
            metadata={},
            complexity_level=ComplexityLevel.SIMPLE
        )
        
        translation_result = TranslationResult(
            original_record_id="test_001",
            translated_question="什麼是Python？",
            translated_answer="Python是一種程式語言",
            translation_strategy="direct",
            terminology_notes=[],
            timestamp="2025-09-11T10:00:00"
        )
        
        qa_result = QAExecutionResult(
            record_id="test_001",
            language=Language.ENGLISH,
            input_question="What is Python?",
            generated_answer="Python is a high-level programming language",
            execution_time=1.5,
            reasoning_steps=[],
            confidence_score=0.9,
            timestamp="2025-09-11T10:01:00"
        )
        
        return ProcessedRecord(
            original_record=original_record,
            translation_result=translation_result,
            original_qa_result=qa_result,
            translated_qa_result=qa_result,
            processing_status=ProcessingStatus.COMPLETED,
            final_quality_score=0.88,
            processing_time=5.0,
            retry_count=0
        )
    
    def test_save_record(self, recovery_manager, sample_record):
        """測試保存記錄"""
        # 保存記錄
        success = recovery_manager.save_record(sample_record)
        assert success
        
        # 檢查文件是否創建
        assert recovery_manager.results_file.exists()
        
        # 檢查內容
        with open(recovery_manager.results_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            record_dict = json.loads(content)
            assert record_dict["id"] == "test_001"
            assert record_dict["processing_status"] == "completed"
    
    def test_get_processed_status_empty(self, recovery_manager):
        """測試空狀態"""
        status = recovery_manager.get_processed_status()
        assert len(status["successful"]) == 0
        assert len(status["failed"]) == 0
        assert status["total"] == 0
    
    def test_get_processed_status_with_records(self, recovery_manager, sample_record):
        """測試有記錄的狀態"""
        # 保存成功記錄
        recovery_manager.save_record(sample_record)
        
        # 保存失敗記錄
        failed_record = ProcessedRecord(
            original_record=sample_record.original_record,
            translation_result=None,
            original_qa_result=None,
            translated_qa_result=None,
            processing_status=ProcessingStatus.FAILED,
            final_quality_score=0.0,
            processing_time=1.0,
            retry_count=1
        )
        recovery_manager.save_record(failed_record)
        
        status = recovery_manager.get_processed_status()
        assert "test_001" in status["successful"]
        assert len(status["failed"]) == 0  # 後來成功了，所以從失敗列表移除
        assert status["total"] == 1
    
    def test_find_missing_and_failed(self, recovery_manager, sample_record):
        """測試找到缺失和失敗記錄"""
        # 保存一筆成功記錄
        recovery_manager.save_record(sample_record)
        
        # 模擬總記錄列表
        all_record_ids = ["test_001", "test_002", "test_003"]
        
        analysis = recovery_manager.find_missing_and_failed(all_record_ids)
        
        assert "test_001" not in analysis["missing"]  # 已處理
        assert "test_002" in analysis["missing"]  # 缺失
        assert "test_003" in analysis["missing"]  # 缺失
        assert len(analysis["need_processing"]) == 2  # 需要處理2筆
    
    def test_load_successful_records(self, recovery_manager, sample_record):
        """測試載入成功記錄"""
        # 保存記錄
        recovery_manager.save_record(sample_record)
        
        # 載入成功記錄
        successful_records = recovery_manager.load_successful_records()
        
        assert len(successful_records) == 1
        assert successful_records[0].original_record.id == "test_001"
        assert successful_records[0].processing_status == ProcessingStatus.COMPLETED
    
    def test_print_status(self, recovery_manager, sample_record, capsys):
        """測試打印狀態"""
        # 保存記錄
        recovery_manager.save_record(sample_record)
        
        # 打印狀態
        all_record_ids = ["test_001", "test_002", "test_003"]
        recovery_manager.print_status(all_record_ids)
        
        captured = capsys.readouterr()
        assert "成功處理: 1 筆" in captured.out
        assert "尚未處理: 2 筆" in captured.out
        assert "需要重跑: 2 筆" in captured.out

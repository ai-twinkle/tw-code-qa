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
        # 先保存失敗記錄
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
        
        # 後來保存成功記錄（覆蓋失敗記錄）
        recovery_manager.save_record(sample_record)
        
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

    def test_incomplete_record_detection(self, recovery_manager):
        """測試不完整記錄檢測"""
        # 創建不完整記錄 - 缺少翻譯答案
        incomplete_record_data = {
            "id": "incomplete_001",
            "original": {
                "question": "Test question",
                "answer": "Test answer",
                "source_dataset": "test",
                "metadata": {},
                "complexity_level": "simple"
            },
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {
                "question": "測試問題",
                "answer": "",  # 空答案
                "strategy": "test",
                "terminology_notes": [],
                "timestamp": 123456789
            },
            "original_qa": {
                "question": "Test question",
                "answer": "Test answer",
                "execution_time": 5.0,
                "reasoning_steps": [],
                "confidence_score": 0.8
            },
            "translated_qa": {
                "question": "測試問題",
                "answer": "",  # 空答案
                "execution_time": 5.0,
                "reasoning_steps": [],
                "confidence_score": 0.8
            }
        }
        
        # 手動寫入不完整記錄
        with open(recovery_manager.results_file, "w", encoding="utf-8") as f:
            json.dump(incomplete_record_data, f, ensure_ascii=False)
            f.write("\n")
        
        # 檢查狀態 - 應該檢測到這是失敗記錄
        status = recovery_manager.get_processed_status()
        assert len(status["successful"]) == 0
        assert "incomplete_001" in status["failed"]
        assert status["total"] == 1

    def test_missing_fields_detection(self, recovery_manager):
        """測試缺失欄位檢測"""
        # 創建缺失必要欄位的記錄
        incomplete_record_data = {
            "id": "missing_fields_001",
            "processing_status": "completed",
            # 缺少 'original' 欄位
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        
        # 手動寫入不完整記錄
        with open(recovery_manager.results_file, "w", encoding="utf-8") as f:
            json.dump(incomplete_record_data, f, ensure_ascii=False)
            f.write("\n")
        
        # 檢查狀態 - 應該檢測到這是失敗記錄
        status = recovery_manager.get_processed_status()
        assert len(status["successful"]) == 0
        assert "missing_fields_001" in status["failed"]
        assert status["total"] == 1

    def test_mixed_complete_incomplete_records(self, recovery_manager, sample_record):
        """測試混合完整和不完整記錄"""
        # 保存完整記錄
        recovery_manager.save_record(sample_record)
        
        # 添加不完整記錄
        incomplete_record_data = {
            "id": "incomplete_002",
            "original": {
                "question": "Test question 2",
                "answer": "Test answer 2",
                "source_dataset": "test",
                "metadata": {},
                "complexity_level": "simple"
            },
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {
                "question": "測試問題2",
                "answer": "測試答案2",
                "strategy": "test",
                "terminology_notes": [],
                "timestamp": 123456789
            },
            "original_qa": {
                "question": "Test question 2",
                "answer": "Test answer 2",
                "execution_time": 5.0,
                "reasoning_steps": [],
                "confidence_score": 0.8
            }
            # 缺少 translated_qa 欄位
        }
        
        # 追加不完整記錄
        with open(recovery_manager.results_file, "a", encoding="utf-8") as f:
            json.dump(incomplete_record_data, f, ensure_ascii=False)
            f.write("\n")
        
        # 檢查狀態
        status = recovery_manager.get_processed_status()
        assert "test_001" in status["successful"]  # 完整記錄
        assert "incomplete_002" in status["failed"]  # 不完整記錄
        assert len(status["successful"]) == 1
        assert len(status["failed"]) == 1
        assert status["total"] == 2

    def test_record_completeness_validation(self, recovery_manager):
        """測試記錄完整性驗證方法"""
        # 測試完整記錄
        complete_record = {
            "id": "complete_001",
            "original": {
                "question": "Test question",
                "answer": "Test answer",
                "source_dataset": "test",
                "metadata": {},
                "complexity_level": "simple"
            },
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {
                "question": "測試問題",
                "answer": "測試答案",
                "strategy": "test",
                "terminology_notes": [],
                "timestamp": 123456789
            },
            "original_qa": {
                "question": "Test question",
                "answer": "Complete answer",
                "execution_time": 5.0,
                "reasoning_steps": [],
                "confidence_score": 0.8
            },
            "translated_qa": {
                "question": "測試問題",
                "answer": "完整答案",
                "execution_time": 5.0,
                "reasoning_steps": [],
                "confidence_score": 0.8
            }
        }
        
        # 測試不完整記錄（空答案）
        incomplete_record = {
            "id": "incomplete_001",
            "original": {
                "question": "Test question",
                "answer": "Test answer",
                "source_dataset": "test",
                "metadata": {},
                "complexity_level": "simple"
            },
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {
                "question": "測試問題",
                "answer": "",  # 空答案
                "strategy": "test",
                "terminology_notes": [],
                "timestamp": 123456789
            },
            "original_qa": {
                "question": "Test question",
                "answer": "Complete answer",
                "execution_time": 5.0,
                "reasoning_steps": [],
                "confidence_score": 0.8
            },
            "translated_qa": {
                "question": "測試問題",
                "answer": "",  # 空答案
                "execution_time": 5.0,
                "reasoning_steps": [],
                "confidence_score": 0.8
            }
        }
        
        # 測試缺失欄位記錄
        missing_field_record = {
            "id": "missing_001",
            "processing_status": "completed",
            # 缺少 'original' 欄位
        }
        
        # 驗證完整性
        assert recovery_manager._is_record_complete(complete_record) == True
        assert recovery_manager._is_record_complete(incomplete_record) == False
        assert recovery_manager._is_record_complete(missing_field_record) == False
    
    def test_save_record_replaces_existing(self, recovery_manager):
        """測試保存記錄會替換現有的同ID記錄"""
        # 創建第一個測試記錄
        record1 = ProcessedRecord(
            original_record=OriginalRecord(
                id="test_replace",
                question="Test question",
                answer="Test answer",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            ),
            translation_result=None,
            original_qa_result=None,
            translated_qa_result=None,
            processing_status=ProcessingStatus.FAILED,
            final_quality_score=5.0,
            processing_time=10.0,
            retry_count=1
        )
        
        # 保存第一個記錄
        success1 = recovery_manager.save_record(record1)
        assert success1 is True
        
        # 創建第二個相同ID但不同內容的記錄
        record2 = ProcessedRecord(
            original_record=OriginalRecord(
                id="test_replace",  # 相同ID
                question="Test question",
                answer="Test answer",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            ),
            translation_result=None,
            original_qa_result=None,
            translated_qa_result=None,
            processing_status=ProcessingStatus.COMPLETED,  # 不同狀態
            final_quality_score=8.0,  # 不同分數
            processing_time=15.0,
            retry_count=2
        )
        
        # 保存第二個記錄（應該替換第一個）
        success2 = recovery_manager.save_record(record2)
        assert success2 is True
        
        # 檢查文件內容
        with open(recovery_manager.results_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 應該只有一行記錄
        assert len(lines) == 1
        
        # 檢查記錄內容是第二個記錄的內容
        record_data = json.loads(lines[0])
        assert record_data["id"] == "test_replace"
        assert record_data["processing_status"] == "completed"
        assert record_data["final_quality_score"] == 8.0
        assert record_data["retry_count"] == 2

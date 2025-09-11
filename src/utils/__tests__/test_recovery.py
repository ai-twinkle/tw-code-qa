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

    def test_save_record_json_decode_error_handling(self, recovery_manager, sample_record):
        """測試保存記錄時處理 JSON 解析錯誤"""
        # 先創建一個包含無效 JSON 的文件
        with open(recovery_manager.results_file, "w", encoding="utf-8") as f:
            f.write('{"id": "existing_001", "invalid": json}\n')  # 無效 JSON
            f.write('{"id": "existing_002", "processing_status": "completed"}\n')  # 有效 JSON
        
        # 保存新記錄，應該跳過無效 JSON 並成功保存
        success = recovery_manager.save_record(sample_record)
        assert success
        
        # 檢查文件內容
        with open(recovery_manager.results_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 應該有 2 行：existing_002 和新的 sample_record
        assert len(lines) == 2
        
        # 檢查新記錄存在
        found_new_record = False
        found_existing_valid = False
        for line in lines:
            record = json.loads(line.strip())
            if record["id"] == "test_001":
                found_new_record = True
            elif record["id"] == "existing_002":
                found_existing_valid = True
        
        assert found_new_record
        assert found_existing_valid

    def test_save_record_general_exception_handling(self, recovery_manager, sample_record):
        """測試保存記錄時處理一般異常"""
        # Mock format converter to raise an exception
        original_converter = recovery_manager.format_converter
        mock_converter = Mock()
        mock_converter.processed_record_to_dict.side_effect = Exception("Mock exception")
        recovery_manager.format_converter = mock_converter
        
        try:
            # 保存記錄應該失敗並記錄錯誤
            success = recovery_manager.save_record(sample_record)
            assert success == False
            
            # 檢查文件沒有被創建或修改
            if recovery_manager.results_file.exists():
                with open(recovery_manager.results_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # 文件應該是空的或只包含舊內容
                    assert "test_001" not in content
        finally:
            # 恢復原始的 format converter
            recovery_manager.format_converter = original_converter

    def test_get_processed_status_json_decode_error_handling(self, recovery_manager):
        """測試獲取處理狀態時處理 JSON 解析錯誤"""
        # 創建包含無效 JSON 的文件
        with open(recovery_manager.results_file, "w", encoding="utf-8") as f:
            f.write('{"id": "valid_001", "processing_status": "completed", "original": {"question": "Q1", "answer": "A1", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"}, "final_quality_score": 8.0, "processing_time": 10.0, "retry_count": 0, "translation": {"question": "TQ1", "answer": "TA1", "strategy": "test", "terminology_notes": [], "timestamp": 123456789}, "original_qa": {"question": "Q1", "answer": "A1", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}, "translated_qa": {"question": "TQ1", "answer": "TA1", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}}\n')
            f.write('{"incomplete": json, "missing": quotes}\n')  # 無效 JSON
            f.write('{"id": "valid_002", "processing_status": "failed", "original": {"question": "Q2", "answer": "A2", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"}, "final_quality_score": 0.0, "processing_time": 1.0, "retry_count": 1}\n')
        
        # 應該能夠處理並跳過無效 JSON
        status = recovery_manager.get_processed_status()
        
        # 應該只統計有效的記錄
        assert "valid_001" in status["successful"]
        assert "valid_002" in status["failed"]
        assert status["total"] == 2

    def test_is_record_complete_missing_basic_fields(self, recovery_manager):
        """測試記錄完整性檢查 - 缺少基本欄位"""
        # 測試缺少 'id' 欄位
        record_no_id = {
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        assert recovery_manager._is_record_complete(record_no_id) == False
        
        # 測試缺少 'original' 欄位
        record_no_original = {
            "id": "test_001",
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        assert recovery_manager._is_record_complete(record_no_original) == False
        
        # 測試缺少 'processing_status' 欄位
        record_no_status = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        assert recovery_manager._is_record_complete(record_no_status) == False

    def test_is_record_complete_missing_original_fields(self, recovery_manager):
        """測試記錄完整性檢查 - 缺少 original 欄位"""
        # 測試缺少 original.question
        record_missing_question = {
            "id": "test_001",
            "original": {"answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        assert recovery_manager._is_record_complete(record_missing_question) == False
        
        # 測試缺少 original.answer
        record_missing_answer = {
            "id": "test_001",
            "original": {"question": "Q", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        assert recovery_manager._is_record_complete(record_missing_answer) == False
        
        # 測試缺少 original.source_dataset
        record_missing_dataset = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        assert recovery_manager._is_record_complete(record_missing_dataset) == False

    def test_is_record_complete_empty_original_fields(self, recovery_manager):
        """測試記錄完整性檢查 - original 欄位為空"""
        # 測試空的 original.question
        record_empty_question = {
            "id": "test_001",
            "original": {"question": "", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        assert recovery_manager._is_record_complete(record_empty_question) == False
        
        # 測試空的 original.answer
        record_empty_answer = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        assert recovery_manager._is_record_complete(record_empty_answer) == False

    def test_is_record_complete_missing_translation_fields(self, recovery_manager):
        """測試記錄完整性檢查 - 缺少翻譯相關欄位"""
        # 測試缺少 translation 欄位
        record_no_translation = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_no_translation) == False
        
        # 測試缺少 translation.question
        record_missing_trans_question = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_missing_trans_question) == False
        
        # 測試空的 translation.question
        record_empty_trans_question = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_empty_trans_question) == False

    def test_is_record_complete_missing_qa_fields(self, recovery_manager):
        """測試記錄完整性檢查 - 缺少 QA 相關欄位"""
        # 測試缺少 original_qa 欄位
        record_no_original_qa = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_no_original_qa) == False
        
        # 測試缺少 translated_qa 欄位
        record_no_translated_qa = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_no_translated_qa) == False

    def test_is_record_complete_invalid_numeric_values(self, recovery_manager):
        """測試記錄完整性檢查 - 無效的數值"""
        # 測試無效的 final_quality_score (負數)
        record_invalid_quality = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": -1.0,  # 無效
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_invalid_quality) == False
        
        # 測試無效的 processing_time (負數)
        record_invalid_time = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": -5.0,  # 無效
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_invalid_time) == False
        
        # 測試無效的 retry_count (負數)
        record_invalid_retry = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": -1,  # 無效
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_invalid_retry) == False

    def test_is_record_complete_invalid_qa_values(self, recovery_manager):
        """測試記錄完整性檢查 - 無效的 QA 數值"""
        # 測試無效的 original_qa.execution_time
        record_invalid_orig_exec_time = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": -1.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_invalid_orig_exec_time) == False
        
        # 測試無效的 confidence_score (超出範圍)
        record_invalid_confidence = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 1.5},  # 超出範圍
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_invalid_confidence) == False

    def test_is_record_complete_invalid_timestamp(self, recovery_manager):
        """測試記錄完整性檢查 - 無效的時間戳"""
        # 測試空的時間戳
        record_empty_timestamp = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": ""},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_empty_timestamp) == False
        
        # 測試 None 時間戳
        record_none_timestamp = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": None},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        assert recovery_manager._is_record_complete(record_none_timestamp) == False

    def test_print_status_success_rate_calculation(self, recovery_manager, sample_record, capsys):
        """測試打印狀態時的成功率計算"""
        # 保存一個成功記錄
        recovery_manager.save_record(sample_record)
        
        # 創建並保存一個失敗記錄
        failed_record = ProcessedRecord(
            original_record=OriginalRecord(
                id="failed_001",
                question="Failed question",
                answer="Failed answer",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            ),
            translation_result=None,
            original_qa_result=None,
            translated_qa_result=None,
            processing_status=ProcessingStatus.FAILED,
            final_quality_score=0.0,
            processing_time=1.0,
            retry_count=1
        )
        recovery_manager.save_record(failed_record)
        
        # 打印狀態（有總記錄數）
        all_record_ids = ["test_001", "failed_001", "missing_001"]
        recovery_manager.print_status(all_record_ids)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # 檢查成功率計算 (1成功 / 2總處理 = 50%)
        assert "成功率: 50.0%" in output
        assert "成功處理: 1 筆" in output
        assert "處理失敗: 1 筆" in output
        assert "總已處理: 2 筆" in output
        assert "尚未處理: 1 筆" in output

    def test_print_status_zero_total_handling(self, recovery_manager, capsys):
        """測試打印狀態時處理總數為0的情況"""
        # 不保存任何記錄，直接打印狀態
        all_record_ids = ["test_001", "test_002"]
        recovery_manager.print_status(all_record_ids)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # 應該沒有成功率顯示，因為總處理數為0
        assert "成功率:" not in output
        assert "成功處理: 0 筆" in output
        assert "尚未處理: 2 筆" in output

    def test_load_successful_records_exception_handling(self, recovery_manager, sample_record):
        """測試載入成功記錄時的異常處理"""
        # 保存一個有效記錄
        recovery_manager.save_record(sample_record)
        
        # 添加無效 JSON 到文件
        with open(recovery_manager.results_file, "a", encoding="utf-8") as f:
            f.write('{"id": "invalid", "broken": json syntax}\n')
            f.write('{"id": "valid_002", "processing_status": "completed", "original": {"question": "Q2", "answer": "A2", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"}, "final_quality_score": 7.0, "processing_time": 8.0, "retry_count": 0}\n')
        
        # 應該能夠處理異常並繼續載入有效的記錄
        successful_records = recovery_manager.load_successful_records()
        
        # 應該只載入有效的記錄
        assert len(successful_records) == 2  # test_001 和 valid_002
        record_ids = {record.original_record.id for record in successful_records}
        assert "test_001" in record_ids
        assert "valid_002" in record_ids

    def test_load_successful_records_general_exception_handling(self, recovery_manager, sample_record):
        """測試載入成功記錄時一般異常的處理"""
        # 保存一個有效記錄
        recovery_manager.save_record(sample_record)
        
        # 添加會導致一般異常的記錄（例如無效的 enum 值）
        with open(recovery_manager.results_file, "a", encoding="utf-8") as f:
            f.write('{"id": "exception_001", "processing_status": "completed", "original": {"question": "Q3", "answer": "A3", "source_dataset": "test", "metadata": {}, "complexity_level": "invalid_enum"}, "final_quality_score": 6.0, "processing_time": 9.0, "retry_count": 0}\n')
        
        # 應該能夠處理一般異常並繼續載入有效的記錄
        successful_records = recovery_manager.load_successful_records()
        
        # 應該只載入有效的記錄，跳過導致異常的記錄
        assert len(successful_records) == 1  # 只有 test_001
        assert successful_records[0].original_record.id == "test_001"

    def test_load_successful_records_file_not_exists(self, recovery_manager):
        """測試載入成功記錄時文件不存在的情況"""
        # 確保文件不存在
        if recovery_manager.results_file.exists():
            recovery_manager.results_file.unlink()
        
        # 應該返回空列表
        successful_records = recovery_manager.load_successful_records()
        assert successful_records == []


class TestRecordCompletenessWarnings:
    """測試記錄完整性檢查的警告訊息"""

    @pytest.fixture
    def recovery_manager(self, tmp_path):
        """創建恢復管理器實例"""
        return RecoveryManager(str(tmp_path))
    
    def test_is_record_complete_warning_messages_missing_basic_fields(self, recovery_manager, caplog):
        """測試缺少基本欄位時的警告訊息"""
        import logging
        
        # 設置日誌級別
        caplog.set_level(logging.WARNING)
        
        # 測試缺少 'id' 欄位
        record_no_id = {
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        
        result = recovery_manager._is_record_complete(record_no_id)
        assert result == False
        
        # 檢查警告訊息
        assert any("missing required field: id" in record.message for record in caplog.records)

    def test_is_record_complete_warning_messages_missing_original_fields(self, recovery_manager, caplog):
        """測試缺少 original 欄位時的警告訊息"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 測試缺少 original.question
        record_missing_question = {
            "id": "test_001",
            "original": {"answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        
        result = recovery_manager._is_record_complete(record_missing_question)
        assert result == False
        
        # 檢查警告訊息
        assert any("missing original.question" in record.message for record in caplog.records)

    def test_is_record_complete_warning_messages_empty_original_fields(self, recovery_manager, caplog):
        """測試 original 欄位為空時的警告訊息"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 測試空的 original.question
        record_empty_question = {
            "id": "test_001",
            "original": {"question": "", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0
        }
        
        result = recovery_manager._is_record_complete(record_empty_question)
        assert result == False
        
        # 檢查警告訊息
        assert any("has empty original.question" in record.message for record in caplog.records)

    def test_is_record_complete_warning_messages_missing_translation_fields(self, recovery_manager, caplog):
        """測試缺少翻譯欄位時的警告訊息"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 測試缺少 translation.question
        record_missing_trans_question = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        
        result = recovery_manager._is_record_complete(record_missing_trans_question)
        assert result == False
        
        # 檢查警告訊息
        assert any("missing translation.question" in record.message for record in caplog.records)

    def test_is_record_complete_warning_messages_empty_translation_fields(self, recovery_manager, caplog):
        """測試翻譯欄位為空時的警告訊息"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 測試空的 translation.question
        record_empty_trans_question = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        
        result = recovery_manager._is_record_complete(record_empty_trans_question)
        assert result == False
        
        # 檢查警告訊息
        assert any("has empty translation.question" in record.message for record in caplog.records)

    def test_is_record_complete_warning_messages_invalid_numeric_values(self, recovery_manager, caplog):
        """測試無效數值時的警告訊息"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 測試無效的 final_quality_score (負數)
        record_invalid_quality = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": -1.0,  # 無效
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        
        result = recovery_manager._is_record_complete(record_invalid_quality)
        assert result == False
        
        # 檢查警告訊息
        assert any("has invalid final_quality_score" in record.message for record in caplog.records)

    def test_is_record_complete_warning_messages_invalid_timestamp(self, recovery_manager, caplog):
        """測試無效時間戳時的警告訊息"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 測試空的時間戳
        record_empty_timestamp = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": ""},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        
        result = recovery_manager._is_record_complete(record_empty_timestamp)
        assert result == False
        
        # 檢查警告訊息
        assert any("has invalid translation.timestamp" in record.message for record in caplog.records)

    def test_is_record_complete_warning_messages_missing_original_qa(self, recovery_manager, caplog):
        """測試缺少 original_qa 欄位時的警告訊息"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 測試缺少 original_qa 欄位
        record_missing_original_qa = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        
        result = recovery_manager._is_record_complete(record_missing_original_qa)
        assert result == False
        
        # 檢查警告訊息
        assert any("missing original_qa" in record.message for record in caplog.records)

    def test_is_record_complete_warning_messages_missing_original_qa_question(self, recovery_manager, caplog):
        """測試缺少 original_qa.question 欄位時的警告訊息"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 測試缺少 original_qa.question
        record_missing_question = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        
        result = recovery_manager._is_record_complete(record_missing_question)
        assert result == False
        
        # 檢查警告訊息
        assert any("missing original_qa.question" in record.message for record in caplog.records)

    def test_is_record_complete_warning_messages_empty_original_qa_question(self, recovery_manager, caplog):
        """測試 original_qa.question 為空時的警告訊息"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 測試空的 original_qa.question
        record_empty_question = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "TQ", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        
        result = recovery_manager._is_record_complete(record_empty_question)
        assert result == False
        
        # 檢查警告訊息
        assert any("has empty original_qa.question" in record.message for record in caplog.records)

    def test_is_record_complete_warning_messages_missing_translation_question(self, recovery_manager, caplog):
        """測試缺少 translation.question 欄位時的警告訊息"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 測試缺少 translation.question
        record_missing_question = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        
        result = recovery_manager._is_record_complete(record_missing_question)
        assert result == False
        
        # 檢查警告訊息
        assert any("missing translation.question" in record.message for record in caplog.records)

    def test_is_record_complete_warning_messages_empty_translation_question(self, recovery_manager, caplog):
        """測試 translation.question 為空時的警告訊息"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 測試空的 translation.question
        record_empty_question = {
            "id": "test_001",
            "original": {"question": "Q", "answer": "A", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"},
            "processing_status": "completed",
            "final_quality_score": 8.5,
            "processing_time": 10.0,
            "retry_count": 0,
            "translation": {"question": "", "answer": "TA", "strategy": "test", "terminology_notes": [], "timestamp": 123456789},
            "original_qa": {"question": "Q", "answer": "A", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8},
            "translated_qa": {"question": "TQ", "answer": "TA", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}
        }
        
        result = recovery_manager._is_record_complete(record_empty_question)
        assert result == False
        
        # 檢查警告訊息
        assert any("has empty translation.question" in record.message for record in caplog.records)


class TestGetProcessedStatusJSONErrorHandling:
    """測試 get_processed_status 的 JSON 錯誤處理"""

    @pytest.fixture
    def recovery_manager(self, tmp_path):
        """創建恢復管理器實例"""
        return RecoveryManager(str(tmp_path))
    
    def test_get_processed_status_json_decode_error_in_loop(self, recovery_manager, caplog):
        """測試處理狀態獲取時迴圈中的 JSON 解析錯誤"""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        # 創建包含無效 JSON 的文件（在有效記錄之間）
        with open(recovery_manager.results_file, "w", encoding="utf-8") as f:
            f.write('{"id": "valid_001", "processing_status": "completed", "original": {"question": "Q1", "answer": "A1", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"}, "final_quality_score": 8.0, "processing_time": 10.0, "retry_count": 0, "translation": {"question": "TQ1", "answer": "TA1", "strategy": "test", "terminology_notes": [], "timestamp": 123456789}, "original_qa": {"question": "Q1", "answer": "A1", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}, "translated_qa": {"question": "TQ1", "answer": "TA1", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}}\n')
            f.write('{"id": "invalid", "broken": json syntax here}\n')  # 無效 JSON
            f.write('{"id": "valid_002", "processing_status": "failed", "original": {"question": "Q2", "answer": "A2", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"}, "final_quality_score": 0.0, "processing_time": 1.0, "retry_count": 1}\n')
        
        # 應該能夠處理並繼續處理其他記錄
        status = recovery_manager.get_processed_status()
        
        # 應該只統計有效的記錄，跳過無效的
        assert "valid_001" in status["successful"]
        assert "valid_002" in status["failed"]
        assert status["total"] == 2  # 不包含無效記錄

    def test_get_processed_status_json_decode_error_syntax_error(self, recovery_manager):
        """測試獲取處理狀態時處理真正無效的 JSON 語法錯誤"""
        # 創建包含真正無效 JSON 語法的文件
        with open(recovery_manager.results_file, "w", encoding="utf-8") as f:
            f.write('{"id": "valid_001", "processing_status": "completed", "original": {"question": "Q1", "answer": "A1", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"}, "final_quality_score": 8.0, "processing_time": 10.0, "retry_count": 0, "translation": {"question": "TQ1", "answer": "TA1", "strategy": "test", "terminology_notes": [], "timestamp": 123456789}, "original_qa": {"question": "Q1", "answer": "A1", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}, "translated_qa": {"question": "TQ1", "answer": "TA1", "execution_time": 5.0, "reasoning_steps": [], "confidence_score": 0.8}}\n')
            f.write('{"incomplete": json, "missing": quotes}\n')  # 真正無效的 JSON 語法
            f.write('{"id": "valid_002", "processing_status": "failed", "original": {"question": "Q2", "answer": "A2", "source_dataset": "test", "metadata": {}, "complexity_level": "simple"}, "final_quality_score": 0.0, "processing_time": 1.0, "retry_count": 1}\n')
        
        # 應該能夠處理並跳過無效 JSON
        status = recovery_manager.get_processed_status()
        
        # 應該只統計有效的記錄
        assert "valid_001" in status["successful"]
        assert "valid_002" in status["failed"]
        assert status["total"] == 2

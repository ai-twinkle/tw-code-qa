"""
測試重寫後的 DatasetManager
Tests for the rewritten DatasetManager
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.dataset_manager import DatasetManager, DatasetProcessingError
from src.models.dataset import OriginalRecord, ProcessedRecord, ProcessingStatus, ComplexityLevel
from src.models.quality import BatchQualityReport


class TestDatasetManager:
    """測試資料集管理器"""
    
    @pytest.fixture
    def temp_dir(self):
        """臨時目錄 fixture"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def dataset_manager(self, temp_dir):
        """資料集管理器 fixture"""
        return DatasetManager(output_dir=temp_dir)
    
    @pytest.fixture
    def sample_record(self):
        """測試記錄 fixture"""
        return OriginalRecord(
            id="test_001",
            question="What is Python?",
            answer="Python is a programming language",
            source_dataset="test_dataset",
            metadata={},
            complexity_level=ComplexityLevel.SIMPLE
        )
    
    def test_initialization(self, temp_dir):
        """測試初始化"""
        manager = DatasetManager(output_dir=temp_dir)
        
        assert manager.output_dir == Path(temp_dir)
        assert manager.enable_checkpointing is True
        assert manager.recovery_manager is not None
        assert manager.workflow_manager is not None
        assert manager.format_converter is not None
    
    def test_process_record_success(self, dataset_manager, sample_record):
        """測試處理單一記錄成功"""
        with patch.object(dataset_manager.workflow_manager, 'process_record') as mock_process:
            # Mock 工作流返回成功狀態
            mock_state = {
                'processing_status': ProcessingStatus.COMPLETED,
                'translation_result': None,
                'original_qa_result': None,
                'translated_qa_result': None,
                'quality_assessment': Mock(overall_quality_score=0.8),
                'retry_count': 0
            }
            mock_process.return_value = mock_state
            
            # 處理記錄
            processed_record = dataset_manager.process_record(sample_record)
            
            # 驗證結果
            assert processed_record.original_record == sample_record
            assert processed_record.processing_status == ProcessingStatus.COMPLETED
            assert processed_record.retry_count == 0
            assert processed_record.processing_time >= 0  # 可能為 0 在快速 mock 情況下
    
    def test_process_record_failure(self, dataset_manager, sample_record):
        """測試處理單一記錄失敗"""
        with patch.object(dataset_manager.workflow_manager, 'process_record') as mock_process:
            # Mock 工作流拋出異常
            mock_process.side_effect = Exception("模擬處理失敗")
            
            # 處理記錄
            processed_record = dataset_manager.process_record(sample_record)
            
            # 驗證結果
            assert processed_record.original_record == sample_record
            assert processed_record.processing_status == ProcessingStatus.FAILED
            assert processed_record.final_quality_score == 0.0
    
    def test_immediate_save(self, dataset_manager, sample_record):
        """測試即時保存功能"""
        with patch.object(dataset_manager.workflow_manager, 'process_record') as mock_process:
            mock_state = {
                'processing_status': ProcessingStatus.COMPLETED,
                'translation_result': None,
                'original_qa_result': None,
                'translated_qa_result': None,
                'quality_assessment': None,
                'retry_count': 0
            }
            mock_process.return_value = mock_state
            
            # 處理並保存記錄
            processed_record = dataset_manager.process_record(sample_record)
            success = dataset_manager.recovery_manager.save_record(processed_record)
            
            assert success is True
            
            # 驗證文件存在
            results_file = dataset_manager.output_dir / "processed_records.jsonl"
            assert results_file.exists()
    
    def test_recovery_functionality(self, dataset_manager):
        """測試恢復功能"""
        test_ids = ["test_001", "test_002", "test_003"]
        
        # 模擬已處理的記錄（只有部分成功）
        with patch.object(dataset_manager.recovery_manager, 'get_processed_status') as mock_status:
            mock_status.return_value = {
                "successful": {"test_001"},
                "failed": {"test_002"},
                "total": 2
            }
            
            # 檢查缺失和失敗記錄
            status = dataset_manager.recovery_manager.find_missing_and_failed(test_ids)
            
            assert "test_003" in status["missing"]  # 未處理
            assert "test_002" in status["failed"]   # 處理失敗
            assert "test_001" not in status["missing"] and "test_001" not in status["failed"]  # 成功
    
    def test_run_with_auto_resume(self, dataset_manager):
        """測試運行功能"""
        # 創建測試記錄
        test_records = [
            OriginalRecord(
                id=f"test_{i:03d}",
                question=f"Question {i}",
                answer=f"Answer {i}",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
            for i in range(1, 3)  # 減少記錄數
        ]
        
        with patch.object(dataset_manager, 'load_dataset') as mock_load:
            mock_load.return_value = test_records
            
            with patch.object(dataset_manager, 'process_record') as mock_process:
                mock_record = Mock(spec=ProcessedRecord)
                mock_record.processing_status = ProcessingStatus.COMPLETED
                mock_process.return_value = mock_record
                
                with patch.object(dataset_manager.recovery_manager, 'save_record') as mock_save:
                    mock_save.return_value = True
                    
                    with patch.object(dataset_manager, '_create_batch_report') as mock_report:
                        mock_batch_report = Mock(spec=BatchQualityReport)
                        mock_report.return_value = mock_batch_report
                        
                        with patch.object(dataset_manager, '_save_final_results'):
                            # 測試運行
                            result = dataset_manager.run("test_path", "opencoder", max_records=2)
                            
                            assert result == mock_batch_report
                            assert mock_load.called
                            assert mock_process.call_count == 2
    
    def test_load_dataset_success(self, dataset_manager):
        """測試載入資料集成功"""
        test_records = [
            OriginalRecord(
                id="test_load",
                question="Test question",
                answer="Test answer",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
        ]
        
        with patch('src.core.dataset_manager.DataLoaderFactory.create_loader') as mock_factory:
            mock_loader = Mock()
            mock_loader.load_dataset.return_value = test_records
            mock_factory.return_value = mock_loader
            
            result = dataset_manager.load_dataset("test_path", "opencoder")
            
            assert len(result) == 1
            assert result[0].id == "test_load"
    
    def test_resume_functionality(self, dataset_manager):
        """測試恢復功能"""
        # 創建測試記錄
        test_records = [
            OriginalRecord(
                id=f"resume_test_{i}",
                question=f"Question {i}",
                answer=f"Answer {i}",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
            for i in range(1, 4)
        ]
        
        with patch.object(dataset_manager, 'load_dataset') as mock_load:
            mock_load.return_value = test_records
            
            with patch.object(dataset_manager.recovery_manager, 'find_missing_and_failed') as mock_find:
                mock_find.return_value = {
                    "missing": {"resume_test_3"},
                    "failed": {"resume_test_2"},
                    "need_processing": {"resume_test_2", "resume_test_3"}
                }
                
                with patch.object(dataset_manager.recovery_manager, 'print_status'):
                    with patch.object(dataset_manager, 'process_record') as mock_process:
                        mock_record = Mock(spec=ProcessedRecord)
                        mock_process.return_value = mock_record
                        
                        with patch.object(dataset_manager.recovery_manager, 'save_record') as mock_save:
                            mock_save.return_value = True
                            
                            with patch.object(dataset_manager.recovery_manager, 'load_successful_records') as mock_load_success:
                                mock_load_success.return_value = []
                                
                                with patch.object(dataset_manager, '_create_batch_report_from_records') as mock_report:
                                    mock_batch_report = Mock(spec=BatchQualityReport)
                                    mock_report.return_value = mock_batch_report
                                    
                                    with patch.object(dataset_manager, '_save_final_results'):
                                        # 測試恢復
                                        result = dataset_manager.resume("test_path")
                                        
                                        assert result == mock_batch_report


class TestDatasetManagerIntegration:
    """整合測試"""
    
    def test_full_workflow_simulation(self):
        """模擬完整工作流程"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(output_dir=temp_dir)
            
            # 創建測試記錄
            test_records = [
                OriginalRecord(
                    id=f"integration_test_{i}",
                    question=f"Integration question {i}",
                    answer=f"Integration answer {i}",
                    source_dataset="integration_test",
                    metadata={},
                    complexity_level=ComplexityLevel.SIMPLE
                )
                for i in range(1, 3)  # 只用少量記錄進行測試
            ]
            
            with patch.object(manager, 'load_dataset') as mock_load:
                mock_load.return_value = test_records
                
                with patch.object(manager.workflow_manager, 'process_record') as mock_process:
                    # Mock 成功處理
                    mock_state = {
                        'processing_status': ProcessingStatus.COMPLETED,
                        'translation_result': None,
                        'original_qa_result': None,
                        'translated_qa_result': None,
                        'quality_assessment': Mock(overall_quality_score=0.8),
                        'retry_count': 0
                    }
                    mock_process.return_value = mock_state
                    
                    with patch.object(manager.recovery_manager, 'save_record') as mock_save:
                        mock_save.return_value = True
                        
                        with patch.object(manager, '_create_batch_report') as mock_report:
                            mock_batch_report = Mock(spec=BatchQualityReport)
                            mock_batch_report.total_records = 2
                            mock_report.return_value = mock_batch_report
                            
                            with patch.object(manager, '_save_final_results'):
                                # 執行處理
                                result = manager.run("test_path", max_records=2)
                                
                                # 驗證結果
                                assert isinstance(result, BatchQualityReport)
                                assert result.total_records == 2

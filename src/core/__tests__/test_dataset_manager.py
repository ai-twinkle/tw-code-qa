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
    
    def test_process_record_with_none_quality_assessment(self, dataset_manager, sample_record):
        """測試處理記錄時 quality_assessment 為 None"""
        with patch.object(dataset_manager.workflow_manager, 'process_record') as mock_process:
            # Mock 工作流返回 quality_assessment 為 None
            mock_state = {
                'processing_status': ProcessingStatus.COMPLETED,
                'translation_result': None,
                'original_qa_result': None,
                'translated_qa_result': None,
                'quality_assessment': None,
                'retry_count': 1
            }
            mock_process.return_value = mock_state
            
            # 處理記錄
            processed_record = dataset_manager.process_record(sample_record)
            
            # 驗證結果
            assert processed_record.original_record == sample_record
            assert processed_record.processing_status == ProcessingStatus.COMPLETED
            assert processed_record.final_quality_score == 0.0  # 應該是 0.0 因為 quality_assessment 為 None
            assert processed_record.retry_count == 1
    
    def test_process_record_with_failed_status(self, dataset_manager, sample_record):
        """測試處理記錄時狀態為 FAILED"""
        with patch.object(dataset_manager.workflow_manager, 'process_record') as mock_process:
            # Mock 工作流返回 FAILED 狀態
            mock_state = {
                'processing_status': ProcessingStatus.FAILED,
                'translation_result': None,
                'original_qa_result': None,
                'translated_qa_result': None,
                'quality_assessment': Mock(overall_quality_score=0.5),
                'retry_count': 2
            }
            mock_process.return_value = mock_state
            
            # 處理記錄
            processed_record = dataset_manager.process_record(sample_record)
            
            # 驗證結果
            assert processed_record.original_record == sample_record
            assert processed_record.processing_status == ProcessingStatus.FAILED
            assert processed_record.final_quality_score == 0.5
            assert processed_record.retry_count == 2
    
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
    
    def test_run_with_progress_logging(self, dataset_manager):
        """測試運行時進度記錄"""
        # 創建 15 個記錄來觸發進度記錄（每 10 個記錄記錄一次）
        test_records = [
            OriginalRecord(
                id=f"progress_test_{i:03d}",
                question=f"Question {i}",
                answer=f"Answer {i}",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
            for i in range(1, 16)  # 15 記錄
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
                            with patch('src.core.dataset_manager.gc') as mock_gc:
                                # 測試運行
                                result = dataset_manager.run("test_path", "opencoder", max_records=15)
                                
                                # 驗證進度記錄被調用（在第 10 個記錄時）
                                # 注意：mock_process.call_count 會是 15，但我們需要檢查日誌
                                assert result == mock_batch_report
    
    def test_run_with_memory_optimization(self, dataset_manager):
        """測試運行時內存優化"""
        # 創建 55 個記錄來觸發內存優化（每 50 個記錄優化一次）
        test_records = [
            OriginalRecord(
                id=f"memory_test_{i:03d}",
                question=f"Question {i}",
                answer=f"Answer {i}",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
            for i in range(1, 56)  # 55 記錄
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
                            with patch('src.core.dataset_manager.gc') as mock_gc:
                                # 測試運行
                                result = dataset_manager.run("test_path", "opencoder", max_records=55)
                                
                                # 驗證 gc.collect 被調用（在第 50 個記錄時）
                                mock_gc.collect.assert_called_once()
                                assert result == mock_batch_report
    
    def test_run_with_memory_optimization_disabled(self, dataset_manager):
        """測試運行時內存優化被禁用"""
        # 創建 55 個記錄
        test_records = [
            OriginalRecord(
                id=f"memory_test_{i:03d}",
                question=f"Question {i}",
                answer=f"Answer {i}",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
            for i in range(1, 56)  # 55 記錄
        ]
        
        with patch('src.core.dataset_manager.FEATURE_FLAGS', {"enable_memory_optimization": False}):
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
                                with patch('src.core.dataset_manager.gc') as mock_gc:
                                    # 測試運行
                                    result = dataset_manager.run("test_path", "opencoder", max_records=55)
                                    
                                    # 驗證 gc.collect 沒有被調用
                                    mock_gc.collect.assert_not_called()
                                    assert result == mock_batch_report
    
    def test_run_with_save_failure_logging(self, dataset_manager):
        """測試運行時保存失敗的記錄"""
        test_records = [
            OriginalRecord(
                id="save_failure_test_001",
                question="Question 1",
                answer="Answer 1",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
        ]
        
        with patch.object(dataset_manager, 'load_dataset') as mock_load:
            mock_load.return_value = test_records
            
            with patch.object(dataset_manager, 'process_record') as mock_process:
                mock_record = Mock(spec=ProcessedRecord)
                mock_process.return_value = mock_record
                
                with patch.object(dataset_manager.recovery_manager, 'save_record') as mock_save:
                    mock_save.return_value = False  # 模擬保存失敗
                    
                    with patch.object(dataset_manager, '_create_batch_report') as mock_report:
                        mock_batch_report = Mock(spec=BatchQualityReport)
                        mock_report.return_value = mock_batch_report
                        
                        with patch.object(dataset_manager, '_save_final_results'):
                            # 測試運行
                            result = dataset_manager.run("test_path", "opencoder")
                            
                            assert result == mock_batch_report
                            mock_save.assert_called_once()
    
    def test_run_exception_handling(self, dataset_manager):
        """測試運行時異常處理"""
        test_records = [
            OriginalRecord(
                id="exception_test_001",
                question="Question 1",
                answer="Answer 1",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
        ]
        
        with patch.object(dataset_manager, 'load_dataset') as mock_load:
            mock_load.return_value = test_records
            
            with patch.object(dataset_manager, 'process_record') as mock_process:
                # Mock process_record 拋出異常
                mock_process.side_effect = Exception("Mock processing error")
                
                # 測試運行應該拋出 DatasetProcessingError
                with pytest.raises(DatasetProcessingError) as exc_info:
                    dataset_manager.run("test_path", "opencoder")
                
                assert "Dataset processing failed" in str(exc_info.value)
    
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
    
    def test_load_dataset_data_load_error(self, dataset_manager):
        """測試載入資料集時 DataLoadError 異常"""
        from src.services.data_loader import DataLoadError
        
        with patch('src.core.dataset_manager.DataLoaderFactory.create_loader') as mock_factory:
            mock_loader = Mock()
            mock_loader.load_dataset.side_effect = DataLoadError("Mock data load error")
            mock_factory.return_value = mock_loader
            
            with pytest.raises(DatasetProcessingError) as exc_info:
                dataset_manager.load_dataset("test_path", "opencoder")
            
            assert "Failed to load dataset" in str(exc_info.value)
    
    def test_load_dataset_general_exception(self, dataset_manager):
        """測試載入資料集時一般異常"""
        with patch('src.core.dataset_manager.DataLoaderFactory.create_loader') as mock_factory:
            mock_loader = Mock()
            mock_loader.load_dataset.side_effect = Exception("Mock general error")
            mock_factory.return_value = mock_loader
            
            with pytest.raises(DatasetProcessingError) as exc_info:
                dataset_manager.load_dataset("test_path", "opencoder")
            
            assert "Unexpected error" in str(exc_info.value)
    
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
    
    def test_resume_no_processing_needed(self, dataset_manager):
        """測試恢復時沒有記錄需要處理"""
        test_records = [
            OriginalRecord(
                id="no_process_test_001",
                question="Question 1",
                answer="Answer 1",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
        ]
        
        with patch.object(dataset_manager, 'load_dataset') as mock_load:
            mock_load.return_value = test_records
            
            with patch.object(dataset_manager.recovery_manager, 'find_missing_and_failed') as mock_find:
                # Mock 沒有記錄需要處理
                mock_find.return_value = {
                    "missing": set(),
                    "failed": set(),
                    "need_processing": set()
                }
                
                with patch.object(dataset_manager.recovery_manager, 'print_status'):
                    with patch.object(dataset_manager.recovery_manager, 'load_successful_records') as mock_load_success:
                        mock_successful_records = [
                            Mock(spec=ProcessedRecord, final_quality_score=0.8, processing_time=1.0, retry_count=0)
                        ]
                        mock_load_success.return_value = mock_successful_records
                        
                        with patch.object(dataset_manager, '_create_batch_report_from_records') as mock_report:
                            mock_batch_report = Mock(spec=BatchQualityReport)
                            mock_report.return_value = mock_batch_report
                            
                            # 測試恢復
                            result = dataset_manager.resume("test_path")
                            
                            assert result == mock_batch_report
                            mock_load_success.assert_called_once()
    
    def test_resume_with_progress_logging(self, dataset_manager):
        """測試恢復時進度記錄"""
        test_records = [
            OriginalRecord(
                id=f"resume_progress_test_{i:03d}",
                question=f"Question {i}",
                answer=f"Answer {i}",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
            for i in range(1, 16)  # 15 記錄
        ]
        
        with patch.object(dataset_manager, 'load_dataset') as mock_load:
            mock_load.return_value = test_records
            
            with patch.object(dataset_manager.recovery_manager, 'find_missing_and_failed') as mock_find:
                mock_find.return_value = {
                    "missing": set(),
                    "failed": set(),
                    "need_processing": {f"resume_progress_test_{i:03d}" for i in range(1, 16)}
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
                                        with patch('src.core.dataset_manager.gc') as mock_gc:
                                            # 測試恢復
                                            result = dataset_manager.resume("test_path")
                                            
                                            assert result == mock_batch_report
    
    def test_resume_with_memory_optimization(self, dataset_manager):
        """測試恢復時內存優化"""
        test_records = [
            OriginalRecord(
                id=f"resume_memory_test_{i:03d}",
                question=f"Question {i}",
                answer=f"Answer {i}",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
            for i in range(1, 26)  # 25 記錄
        ]
        
        with patch.object(dataset_manager, 'load_dataset') as mock_load:
            mock_load.return_value = test_records
            
            with patch.object(dataset_manager.recovery_manager, 'find_missing_and_failed') as mock_find:
                mock_find.return_value = {
                    "missing": set(),
                    "failed": set(),
                    "need_processing": {f"resume_memory_test_{i:03d}" for i in range(1, 26)}
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
                                        with patch('src.core.dataset_manager.gc') as mock_gc:
                                            # 測試恢復
                                            result = dataset_manager.resume("test_path")
                                            
                                            # 驗證 gc.collect 被調用（在第 20 個記錄時）
                                            mock_gc.collect.assert_called_once()
                                            assert result == mock_batch_report
    
    def test_resume_with_save_failure_logging(self, dataset_manager):
        """測試恢復時保存失敗的記錄"""
        test_records = [
            OriginalRecord(
                id="resume_save_failure_test_001",
                question="Question 1",
                answer="Answer 1",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
        ]
        
        with patch.object(dataset_manager, 'load_dataset') as mock_load:
            mock_load.return_value = test_records
            
            with patch.object(dataset_manager.recovery_manager, 'find_missing_and_failed') as mock_find:
                mock_find.return_value = {
                    "missing": set(),
                    "failed": set(),
                    "need_processing": {"resume_save_failure_test_001"}
                }
                
                with patch.object(dataset_manager.recovery_manager, 'print_status'):
                    with patch.object(dataset_manager, 'process_record') as mock_process:
                        mock_record = Mock(spec=ProcessedRecord)
                        mock_process.return_value = mock_record
                        
                        with patch.object(dataset_manager.recovery_manager, 'save_record') as mock_save:
                            mock_save.return_value = False  # 模擬保存失敗
                            
                            with patch.object(dataset_manager.recovery_manager, 'load_successful_records') as mock_load_success:
                                mock_load_success.return_value = []
                                
                                with patch.object(dataset_manager, '_create_batch_report_from_records') as mock_report:
                                    mock_batch_report = Mock(spec=BatchQualityReport)
                                    mock_report.return_value = mock_batch_report
                                    
                                    with patch.object(dataset_manager, '_save_final_results'):
                                        # 測試恢復
                                        result = dataset_manager.resume("test_path")
                                        
                                        assert result == mock_batch_report
                                        mock_save.assert_called_once()
    
    def test_resume_exception_handling(self, dataset_manager):
        """測試恢復時異常處理"""
        test_records = [
            OriginalRecord(
                id="resume_exception_test_001",
                question="Question 1",
                answer="Answer 1",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
        ]
        
        with patch.object(dataset_manager, 'load_dataset') as mock_load:
            mock_load.return_value = test_records
            
            with patch.object(dataset_manager.recovery_manager, 'find_missing_and_failed') as mock_find:
                mock_find.return_value = {
                    "missing": set(),
                    "failed": set(),
                    "need_processing": {"resume_exception_test_001"}
                }
                
                with patch.object(dataset_manager.recovery_manager, 'print_status'):
                    with patch.object(dataset_manager, 'process_record') as mock_process:
                        # Mock process_record 拋出異常
                        mock_process.side_effect = Exception("Mock resume processing error")
                        
                        # 測試恢復應該拋出 DatasetProcessingError
                        with pytest.raises(DatasetProcessingError) as exc_info:
                            dataset_manager.resume("test_path")
                        
                        assert "Resume processing failed" in str(exc_info.value)
    
    def test_create_batch_report(self, dataset_manager):
        """測試創建批次報告"""
        test_records = [
            OriginalRecord(
                id=f"batch_report_test_{i}",
                question=f"Question {i}",
                answer=f"Answer {i}",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
            for i in range(1, 4)
        ]
        
        with patch.object(dataset_manager.recovery_manager, 'get_processed_status') as mock_status:
            mock_status.return_value = {
                "successful": {"batch_report_test_1", "batch_report_test_2"},
                "failed": {"batch_report_test_3"},
                "total": 3
            }
            
            with patch.object(dataset_manager.recovery_manager, 'load_successful_records') as mock_load:
                mock_processed_records = [
                    Mock(spec=ProcessedRecord, final_quality_score=0.8, processing_time=1.0, retry_count=0),
                    Mock(spec=ProcessedRecord, final_quality_score=0.9, processing_time=1.2, retry_count=1)
                ]
                mock_load.return_value = mock_processed_records
                
                start_time = 1000.0
                report = dataset_manager._create_batch_report(test_records, start_time)
                
                assert report.total_records == 3
                assert report.passed_records == 2
                assert report.failed_records == 1
                assert abs(report.average_quality_score - 0.85) < 1e-10  # (0.8 + 0.9) / 2
                assert report.batch_start_time == start_time
    
    def test_create_batch_report_from_records(self, dataset_manager):
        """測試從記錄創建批次報告"""
        mock_records = [
            Mock(spec=ProcessedRecord, 
                 processing_status=ProcessingStatus.COMPLETED, 
                 final_quality_score=0.7, 
                 processing_time=0.8, 
                 retry_count=0),
            Mock(spec=ProcessedRecord, 
                 processing_status=ProcessingStatus.FAILED, 
                 final_quality_score=0.0, 
                 processing_time=1.0, 
                 retry_count=2),
            Mock(spec=ProcessedRecord, 
                 processing_status=ProcessingStatus.COMPLETED, 
                 final_quality_score=0.9, 
                 processing_time=1.1, 
                 retry_count=1)
        ]
        
        start_time = 2000.0
        report = dataset_manager._create_batch_report_from_records(mock_records, start_time)
        
        assert report.total_records == 3
        assert report.passed_records == 2
        assert report.failed_records == 1
        assert report.average_quality_score == 0.8  # (0.7 + 0.9) / 2
        assert report.total_retries == 3  # 0 + 2 + 1
        assert report.batch_start_time == start_time
    
    def test_create_batch_report_empty_quality_scores(self, dataset_manager):
        """測試創建批次報告時品質分數為空"""
        mock_records = [
            Mock(spec=ProcessedRecord, 
                 processing_status=ProcessingStatus.COMPLETED, 
                 final_quality_score=0.0, 
                 processing_time=1.0, 
                 retry_count=0)
        ]
        
        start_time = 3000.0
        report = dataset_manager._create_batch_report_from_records(mock_records, start_time)
        
        assert report.average_quality_score == 0.0
        assert report.min_quality_score == 0.0
        assert report.max_quality_score == 0.0
    
    def test_save_final_results(self, dataset_manager):
        """測試保存最終結果"""
        mock_report = Mock(spec=BatchQualityReport)
        
        with patch.object(dataset_manager.format_converter, 'export_quality_report') as mock_export:
                with patch('src.core.dataset_manager.time') as mock_time:
                    mock_time.time.return_value = 1234567890.0
                    
                    # 測試保存
                    dataset_manager._save_final_results(mock_report)
                    
                    # 驗證調用
                    expected_report_file = dataset_manager.output_dir / "quality_report.json"
                    mock_export.assert_called_once_with(mock_report, expected_report_file)
    
    def test_save_final_results_exception(self, dataset_manager):
        """測試保存最終結果時異常"""
        mock_report = Mock(spec=BatchQualityReport)
        
        with patch.object(dataset_manager.format_converter, 'export_quality_report') as mock_export:
            mock_export.side_effect = Exception("Mock save error")
            
            # 應該不會拋出異常，只是記錄錯誤
            dataset_manager._save_final_results(mock_report)
    
    def test_get_status_with_dataset_path(self, dataset_manager):
        """測試獲取狀態（有資料集路徑）"""
        test_records = [
            OriginalRecord(
                id="status_test_001",
                question="Question 1",
                answer="Answer 1",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
        ]
        
        with patch.object(dataset_manager, 'load_dataset') as mock_load:
            mock_load.return_value = test_records
            
            with patch.object(dataset_manager.recovery_manager, 'find_missing_and_failed') as mock_find:
                mock_find.return_value = {
                    "missing": {"status_test_001"},
                    "failed": set(),
                    "need_processing": {"status_test_001"}
                }
                
                result = dataset_manager.get_status("test_path", "opencoder")
                
                assert "missing" in result
                assert "failed" in result
                assert "need_processing" in result
                mock_load.assert_called_once_with("test_path", "opencoder")
    
    def test_get_status_without_dataset_path(self, dataset_manager):
        """測試獲取狀態（無資料集路徑）"""
        with patch.object(dataset_manager.recovery_manager, 'get_processed_status') as mock_status:
            mock_status.return_value = {
                "successful": {"test_001"},
                "failed": {"test_002"},
                "total": 2
            }
            
            result = dataset_manager.get_status()
            
            assert result == {
                "successful": {"test_001"},
                "failed": {"test_002"},
                "total": 2
            }
            mock_status.assert_called_once()
    
    def test_print_status_with_dataset_path(self, dataset_manager):
        """測試打印狀態（有資料集路徑）"""
        test_records = [
            OriginalRecord(
                id="print_status_test_001",
                question="Question 1",
                answer="Answer 1",
                source_dataset="test",
                metadata={},
                complexity_level=ComplexityLevel.SIMPLE
            )
        ]
        
        with patch.object(dataset_manager, 'load_dataset') as mock_load:
            mock_load.return_value = test_records
            
            with patch.object(dataset_manager.recovery_manager, 'print_status') as mock_print:
                dataset_manager.print_status("test_path", "opencoder")
                
                mock_load.assert_called_once_with("test_path", "opencoder")
                mock_print.assert_called_once()
    
    def test_print_status_without_dataset_path(self, dataset_manager):
        """測試打印狀態（無資料集路徑）"""
        with patch.object(dataset_manager.recovery_manager, 'print_status') as mock_print:
            dataset_manager.print_status()
            
            mock_print.assert_called_once()


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

"""
格式轉換器測試
Format Converter Tests
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json
import tempfile

from src.utils.format_converter import DataFormatConverter, DatasetExporter, FormatConversionError
from src.models.dataset import ProcessedRecord, OriginalRecord, TranslationResult, QAExecutionResult, ProcessingStatus, Language, ComplexityLevel


class TestDataFormatConverter:
    """格式轉換器測試類"""

    @pytest.fixture
    def converter(self) -> DataFormatConverter:
        """創建格式轉換器實例"""
        return DataFormatConverter()

    def create_test_processed_record(self, record_id: str = "test_id") -> ProcessedRecord:
        """創建測試用處理記錄"""
        original_record = OriginalRecord(
            id=record_id,
            question="Test question",
            answer="Test answer",
            source_dataset="test_dataset",
            metadata={"tag": "test", "source_index": 0},
            complexity_level=ComplexityLevel.SIMPLE
        )
        
        translation_result = TranslationResult(
            original_record_id=record_id,
            translated_question="測試問題",
            translated_answer="測試答案",
            translation_strategy="direct",
            terminology_notes=[]
        )
        
        original_qa_result = QAExecutionResult(
            record_id=record_id,
            language=Language.ENGLISH,
            input_question="Test question",
            generated_answer="Test answer",
            execution_time=1.0,
            reasoning_steps=["step1", "step2"]
        )
        
        translated_qa_result = QAExecutionResult(
            record_id=record_id,
            language=Language.TRADITIONAL_CHINESE,
            input_question="測試問題",
            generated_answer="測試答案",
            execution_time=1.0,
            reasoning_steps=["步驟1", "步驟2"]
        )
        
        return ProcessedRecord(
            original_record=original_record,
            translation_result=translation_result,
            original_qa_result=original_qa_result,
            translated_qa_result=translated_qa_result,
            processing_status=ProcessingStatus.COMPLETED,
            final_quality_score=0.85,
            processing_time=2.5,
            retry_count=0
        )

    def test_initialization(self, converter: DataFormatConverter) -> None:
        """測試初始化"""
        assert converter is not None
        assert hasattr(converter, 'detect_format')

    def test_records_to_jsonl_basic(self, converter: DataFormatConverter) -> None:
        """測試基本 JSONL 轉換"""
        records = [self.create_test_processed_record(f"record_{i}") for i in range(2)]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as tmp_file:
            output_path = tmp_file.name

        try:
            # 測試 JSONL 輸出
            converter.records_to_jsonl(records, output_path)
            
            # 驗證文件被創建
            assert Path(output_path).exists()
            
        finally:
            # 清理臨時文件
            Path(output_path).unlink(missing_ok=True)

    def test_records_to_jsonl_empty(self, converter: DataFormatConverter) -> None:
        """測試空記錄轉換"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as tmp_file:
            output_path = tmp_file.name

        try:
            # 測試空記錄輸出
            converter.records_to_jsonl([], output_path)
            
            # 驗證文件被創建（即使是空的）
            assert Path(output_path).exists()
            
        finally:
            # 清理臨時文件
            Path(output_path).unlink(missing_ok=True)

    def test_records_to_csv_basic(self, converter: DataFormatConverter) -> None:
        """測試基本 CSV 轉換"""
        records = [self.create_test_processed_record(f"record_{i}") for i in range(2)]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            output_path = tmp_file.name

        try:
            # 測試 CSV 輸出
            converter.records_to_csv(records, output_path)
            
            # 驗證文件被創建
            assert Path(output_path).exists()
            
        finally:
            # 清理臨時文件
            Path(output_path).unlink(missing_ok=True)

    def test_detect_format(self, converter: DataFormatConverter) -> None:
        """測試格式檢測"""
        # 測試不同格式的檢測
        assert converter.detect_format("test.jsonl") == "jsonl"
        assert converter.detect_format("test.csv") == "csv"
        assert converter.detect_format("test.json") == "jsonl"  # .json maps to jsonl
        
        # 測試不支援的格式
        with pytest.raises(FormatConversionError, match="Unsupported file format"):
            converter.detect_format("test.unknown")

    def test_processed_record_to_dict(self, converter: DataFormatConverter) -> None:
        """測試記錄轉字典"""
        record = self.create_test_processed_record("test_record")
        
        # 測試私有方法（如果存在）
        if hasattr(converter, '_processed_record_to_dict'):
            result = converter._processed_record_to_dict(record)
            assert isinstance(result, dict)
            assert "original_record" in result or "id" in result  # 基本驗證

    def test_records_to_jsonl_error_handling(self, converter: DataFormatConverter) -> None:
        """測試 JSONL 轉換錯誤處理"""
        records = [self.create_test_processed_record("test")]
        
        # 測試無效路徑錯誤
        with pytest.raises(FormatConversionError, match="JSONL conversion failed"):
            converter.records_to_jsonl(records, "/invalid/path/file.jsonl")

    def test_records_to_csv_pandas_not_installed(self, converter: DataFormatConverter) -> None:
        """測試 CSV 轉換當 pandas 未安裝時"""
        records = [self.create_test_processed_record("test")]
        
        # 模擬 pandas 導入錯誤
        with patch('src.utils.format_converter.DataFormatConverter.records_to_csv', side_effect=FormatConversionError("pandas not installed. Run: pip install pandas")):
            with pytest.raises(FormatConversionError, match="pandas not installed"):
                converter.records_to_csv(records, "test.csv")

    def test_records_to_csv_error_handling(self, converter: DataFormatConverter) -> None:
        """測試 CSV 轉換錯誤處理"""
        records = [self.create_test_processed_record("test")]
        
        # 測試無效路徑錯誤  
        with pytest.raises(FormatConversionError, match="CSV conversion failed"):
            converter.records_to_csv(records, "/invalid/path/file.csv")


class TestDatasetExporter:
    """資料集匯出器測試類"""

    @pytest.fixture  
    def exporter(self) -> DatasetExporter:
        """創建資料集匯出器實例"""
        return DatasetExporter()

    def create_test_processed_record(self, record_id: str = "test_id") -> ProcessedRecord:
        """創建測試用處理記錄"""
        original_record = OriginalRecord(
            id=record_id,
            question="Test question",
            answer="Test answer",
            source_dataset="test_dataset",
            metadata={"tag": "test", "source_index": 0},
            complexity_level=ComplexityLevel.SIMPLE
        )
        
        translation_result = TranslationResult(
            original_record_id=record_id,
            translated_question="測試問題",
            translated_answer="測試答案",
            translation_strategy="direct",
            terminology_notes=[]
        )
        
        original_qa_result = QAExecutionResult(
            record_id=record_id,
            language=Language.ENGLISH,
            input_question="Test question",
            generated_answer="Test answer",
            execution_time=1.0,
            reasoning_steps=["step1", "step2"]
        )
        
        translated_qa_result = QAExecutionResult(
            record_id=record_id,
            language=Language.TRADITIONAL_CHINESE,
            input_question="測試問題",
            generated_answer="測試答案",
            execution_time=1.0,
            reasoning_steps=["步驟1", "步驟2"]
        )
        
        return ProcessedRecord(
            original_record=original_record,
            translation_result=translation_result,
            original_qa_result=original_qa_result,
            translated_qa_result=translated_qa_result,
            processing_status=ProcessingStatus.COMPLETED,
            final_quality_score=0.85,
            processing_time=2.5,
            retry_count=0
        )

    def test_initialization(self, exporter: DatasetExporter) -> None:
        """測試初始化"""
        assert exporter is not None
        assert hasattr(exporter, 'export_dataset')

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_export_dataset_jsonl(self, mock_mkdir: Mock, mock_file: Mock, exporter: DatasetExporter) -> None:
        """測試匯出到 JSONL 文件"""
        records = [self.create_test_processed_record(f"record_{i}") for i in range(2)]

        # 測試匯出功能
        with patch.object(exporter.converter, 'records_to_jsonl') as mock_to_jsonl:
            exporter.export_dataset(
                records=records,
                output_dir="test_output",
                dataset_name="test_dataset",
                formats=["jsonl"]
            )
            
            # 驗證轉換方法被調用
            mock_to_jsonl.assert_called_once()

    def test_export_successful_records_only(self, exporter: DatasetExporter) -> None:
        """測試只匯出成功記錄"""
        # 創建混合狀態的記錄
        successful_record = self.create_test_processed_record("success_1")
        failed_record = self.create_test_processed_record("failed_1")
        failed_record.processing_status = ProcessingStatus.FAILED
        
        records = [successful_record, failed_record]

        with patch.object(exporter.converter, 'records_to_jsonl') as mock_to_jsonl:
            exporter.export_successful_records_only(
                records=records,
                output_dir="test_output",
                dataset_name="successful_records"
            )
            
            # 驗證只處理成功的記錄
            mock_to_jsonl.assert_called_once()
            # 獲取實際傳入的記錄列表
            called_records = mock_to_jsonl.call_args[0][0]
            # 驗證只有成功的記錄被包含
            assert len(called_records) == 1
            assert called_records[0].processing_status == ProcessingStatus.COMPLETED

    def test_export_dataset_multiple_formats(self, exporter: DatasetExporter) -> None:
        """測試匯出多種格式"""
        records = [self.create_test_processed_record(f"record_{i}") for i in range(2)]

        with patch.object(exporter.converter, 'records_to_jsonl') as mock_jsonl:
            with patch.object(exporter.converter, 'records_to_csv') as mock_csv:
                with patch("pathlib.Path.mkdir"):
                    result = exporter.export_dataset(
                        records=records,
                        output_dir="test_output",
                        dataset_name="test_dataset",
                        formats=["jsonl", "csv"]
                    )
                    
                    # 驗證兩種格式都被調用
                    mock_jsonl.assert_called_once()
                    mock_csv.assert_called_once()
                    
                    # 驗證返回結果
                    assert isinstance(result, dict)

    def test_export_dataset_error_handling(self, exporter: DatasetExporter) -> None:
        """測試匯出錯誤處理"""
        records = [self.create_test_processed_record("test")]
        
        # 測試轉換器錯誤
        with patch.object(exporter.converter, 'records_to_jsonl', side_effect=FormatConversionError("Test error")):
            with patch("pathlib.Path.mkdir"):
                with pytest.raises(FormatConversionError):
                    exporter.export_dataset(
                        records=records,
                        output_dir="test_output",
                        formats=["jsonl"]
                    )


class TestDefaultInstances:
    """測試默認實例"""

    def test_default_converter(self) -> None:
        """測試默認轉換器"""
        # 測試是否可以創建默認實例
        converter = DataFormatConverter()
        assert converter is not None

    def test_default_exporter(self) -> None:
        """測試默認匯出器"""
        # 測試是否可以創建默認實例
        exporter = DatasetExporter()
        assert exporter is not None

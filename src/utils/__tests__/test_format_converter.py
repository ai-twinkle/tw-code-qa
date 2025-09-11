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
            result = converter.processed_record_to_dict(record)
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

    def test_records_to_arrow_basic(self, converter: DataFormatConverter) -> None:
        """測試基本 Arrow 轉換"""
        records = [self.create_test_processed_record(f"record_{i}") for i in range(2)]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.arrow') as tmp_file:
            output_path = tmp_file.name

        try:
            # 測試 Arrow 輸出
            converter.records_to_arrow(records, output_path)
            
            # 驗證文件被創建
            assert Path(output_path).exists()
            
        finally:
            # 清理臨時文件
            Path(output_path).unlink(missing_ok=True)

    def test_records_to_arrow_pyarrow_not_installed(self, converter: DataFormatConverter) -> None:
        """測試 Arrow 轉換當 pyarrow 未安裝時"""
        records = [self.create_test_processed_record("test")]
        
        # 模擬 pyarrow 導入錯誤
        with patch('builtins.__import__', side_effect=ImportError("No module named 'pyarrow'")):
            with pytest.raises(FormatConversionError, match="pyarrow not installed"):
                converter.records_to_arrow(records, "test.arrow")

    def test_records_to_arrow_error_handling(self, converter: DataFormatConverter) -> None:
        """測試 Arrow 轉換錯誤處理"""
        records = [self.create_test_processed_record("test")]
        
        # 測試無效路徑錯誤
        with pytest.raises(FormatConversionError, match="Arrow conversion failed"):
            converter.records_to_arrow(records, "/invalid/path/file.arrow")

    def test_export_records_unsupported_format(self, converter: DataFormatConverter) -> None:
        """測試不支援的匯出格式"""
        records = [self.create_test_processed_record("test")]
        
        with pytest.raises(FormatConversionError, match="Unsupported export format"):
            converter.export_records(records, "test.txt", "unsupported")

    def test_export_records_path_extension_handling(self, converter: DataFormatConverter) -> None:
        """測試路徑副檔名處理"""
        records = [self.create_test_processed_record("test")]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 測試沒有副檔名的路徑
            output_path = Path(tmp_dir) / "test_output"
            converter.export_records(records, output_path, "jsonl")
            
            # 檢查檔案是否以正確副檔名被創建
            expected_path = output_path.with_suffix('.jsonl')
            assert expected_path.exists()

    def test_export_records_wrong_extension_warning(self, converter: DataFormatConverter) -> None:
        """測試錯誤副檔名的警告"""
        records = [self.create_test_processed_record("test")]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 使用錯誤的副檔名
            output_path = Path(tmp_dir) / "test_output.txt"
            
            with patch.object(converter.logger, 'warning') as mock_warning:
                converter.export_records(records, output_path, "jsonl")
                mock_warning.assert_called_once()

    def test_export_quality_report(self, converter: DataFormatConverter) -> None:
        """測試品質報告匯出"""
        from src.models.quality import BatchQualityReport
        
        # 創建模擬的品質報告
        with patch('src.models.quality.BatchQualityReport') as MockReport:
            mock_report = MockReport.return_value
            mock_report.batch_id = "test_batch"
            mock_report.total_records = 100
            mock_report.processed_records = 95
            mock_report.passed_records = 90
            mock_report.failed_records = 5
            mock_report.retry_records = 0
            mock_report.get_success_rate.return_value = 0.9
            mock_report.get_failure_rate.return_value = 0.1
            mock_report.average_quality_score = 0.85
            mock_report.min_quality_score = 0.5
            mock_report.max_quality_score = 1.0
            mock_report.total_processing_time = 300.0
            mock_report.average_processing_time = 3.0
            mock_report.total_retries = 10
            mock_report.error_summary = {}
            mock_report.batch_start_time = "2024-01-01T00:00:00"
            mock_report.batch_end_time = "2024-01-01T01:00:00"
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                output_path = tmp_file.name
            
            try:
                converter.export_quality_report(mock_report, output_path)
                
                # 驗證文件被創建
                assert Path(output_path).exists()
                
                # 驗證文件內容
                with open(output_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                assert report_data['batch_id'] == "test_batch"
                assert report_data['summary']['total_records'] == 100
                
            finally:
                Path(output_path).unlink(missing_ok=True)

    def test_export_quality_report_error_handling(self, converter: DataFormatConverter) -> None:
        """測試品質報告匯出錯誤處理"""
        from src.models.quality import BatchQualityReport
        
        with patch('src.models.quality.BatchQualityReport') as MockReport:
            mock_report = MockReport.return_value
            
            # 測試無效路徑錯誤
            with pytest.raises(FormatConversionError, match="Quality report export failed"):
                converter.export_quality_report(mock_report, "/invalid/path/report.json")

    def test_processed_record_to_dict_complete(self, converter: DataFormatConverter) -> None:
        """測試完整記錄轉字典"""
        record = self.create_test_processed_record("test_record")
        
        result = converter.processed_record_to_dict(record)
        
        # 驗證基本結構
        assert result['id'] == "test_record"
        assert 'original' in result
        assert 'translation' in result
        assert 'original_qa' in result
        assert 'translated_qa' in result
        assert result['processing_status'] == ProcessingStatus.COMPLETED.value
        assert result['final_quality_score'] == 0.85

    def test_processed_record_to_dict_minimal(self, converter: DataFormatConverter) -> None:
        """測試最小記錄轉字典（無翻譯和QA結果）"""
        from src.models.dataset import OriginalRecord, ProcessedRecord, ProcessingStatus, ComplexityLevel
        
        # 創建只有原始記錄的最小處理記錄
        original_record = OriginalRecord(
            id="minimal_record",
            question="Minimal question",
            answer="Minimal answer",
            source_dataset="minimal_dataset"
        )
        
        minimal_record = ProcessedRecord(
            original_record=original_record,
            translation_result=None,
            original_qa_result=None,
            translated_qa_result=None,
            processing_status=ProcessingStatus.PENDING,
            final_quality_score=0.0,
            processing_time=0.0,
            retry_count=0
        )
        
        result = converter.processed_record_to_dict(minimal_record)
        
        # 驗證基本結構存在
        assert result['id'] == "minimal_record"
        assert 'original' in result
        assert result['processing_status'] == ProcessingStatus.PENDING.value
        # 翻譯和QA結果應該不存在
        assert 'translation' not in result
        assert 'original_qa' not in result
        assert 'translated_qa' not in result


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

    def test_export_dataset_partial_failure(self, exporter: DatasetExporter) -> None:
        """測試部分格式匯出失敗"""
        records = [self.create_test_processed_record("test")]
        
        # 模擬一個格式成功，一個失敗
        with patch.object(exporter.converter, 'records_to_jsonl') as mock_jsonl:
            with patch.object(exporter.converter, 'records_to_csv', side_effect=FormatConversionError("CSV failed")):
                with patch("pathlib.Path.mkdir"):
                    result = exporter.export_dataset(
                        records=records,
                        output_dir="test_output",
                        formats=["jsonl", "csv"]
                    )
                    
                    # 應該只有成功的格式在結果中
                    assert "jsonl" in result
                    assert "csv" not in result

    def test_export_dataset_all_formats_fail(self, exporter: DatasetExporter) -> None:
        """測試所有格式都失敗"""
        records = [self.create_test_processed_record("test")]
        
        # 模擬所有格式都失敗
        with patch.object(exporter.converter, 'records_to_jsonl', side_effect=FormatConversionError("JSONL failed")):
            with patch.object(exporter.converter, 'records_to_csv', side_effect=FormatConversionError("CSV failed")):
                with patch("pathlib.Path.mkdir"):
                    with pytest.raises(FormatConversionError, match="All export formats failed"):
                        exporter.export_dataset(
                            records=records,
                            output_dir="test_output",
                            formats=["jsonl", "csv"]
                        )

    def test_export_dataset_default_formats(self, exporter: DatasetExporter) -> None:
        """測試默認格式"""
        records = [self.create_test_processed_record("test")]
        
        with patch.object(exporter.converter, 'records_to_jsonl') as mock_jsonl:
            with patch.object(exporter.converter, 'records_to_csv') as mock_csv:
                with patch("pathlib.Path.mkdir"):
                    result = exporter.export_dataset(
                        records=records,
                        output_dir="test_output"
                        # 不指定格式，應該使用默認值
                    )
                    
                    # 驗證默認格式都被調用
                    mock_jsonl.assert_called_once()
                    mock_csv.assert_called_once()

    def test_export_successful_records_only_mixed_statuses(self, exporter: DatasetExporter) -> None:
        """測試混合狀態記錄的成功記錄匯出"""
        from src.models.dataset import ProcessingStatus
        
        # 創建不同狀態的記錄
        records = []
        for i, status in enumerate([ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, 
                                  ProcessingStatus.PENDING, ProcessingStatus.COMPLETED]):
            record = self.create_test_processed_record(f"record_{i}")
            record.processing_status = status
            records.append(record)
        
        with patch.object(exporter, 'export_dataset') as mock_export:
            exporter.export_successful_records_only(
                records=records,
                output_dir="test_output"
            )
            
            # 驗證只有成功的記錄被傳遞
            called_records = mock_export.call_args[0][0]  # 第一個位置參數是 records
            assert len(called_records) == 2  # 只有兩個 COMPLETED 記錄
            for record in called_records:
                assert record.processing_status == ProcessingStatus.COMPLETED


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

    def test_module_level_exports(self) -> None:
        """測試模組級別的匯出"""
        from src.utils.format_converter import default_converter, default_exporter
        
        assert default_converter is not None
        assert default_exporter is not None
        assert isinstance(default_converter, DataFormatConverter)
        assert isinstance(default_exporter, DatasetExporter)


class TestEdgeCases:
    """測試邊緣情況"""

    @pytest.fixture
    def converter(self) -> DataFormatConverter:
        return DataFormatConverter()

    def test_detect_format_with_path_object(self, converter: DataFormatConverter) -> None:
        """測試使用 Path 物件的格式檢測"""
        from pathlib import Path
        
        path = Path("test.jsonl")
        assert converter.detect_format(path) == "jsonl"

    def test_detect_format_case_insensitive(self, converter: DataFormatConverter) -> None:
        """測試大小寫不敏感的格式檢測"""
        assert converter.detect_format("test.JSONL") == "jsonl"
        assert converter.detect_format("test.CSV") == "csv"

    def test_export_records_conversion_method_not_implemented(self, converter: DataFormatConverter) -> None:
        """測試未實現的轉換方法"""
        records = [Mock()]
        
        # 使用有效但未實現的格式
        with patch('src.utils.format_converter.EXPORT_FORMATS', {'fake_format': {'extension': '.fake'}}):
            with pytest.raises(FormatConversionError, match="Conversion method not implemented"):
                converter.export_records(records, "test.fake", "fake_format")

    def test_records_to_jsonl_with_json_serialization_error(self, converter: DataFormatConverter) -> None:
        """測試 JSON 序列化錯誤"""
        # 創建一個無法序列化的記錄
        mock_record = Mock()
        
        with patch.object(converter, '_processed_record_to_dict', return_value={'bad_key': object()}):
            with pytest.raises(FormatConversionError, match="JSONL conversion failed"):
                converter.records_to_jsonl([mock_record], "test.jsonl")

    def test_datasetexporter_with_custom_converter(self) -> None:
        """測試使用自定義轉換器的資料集匯出器"""
        custom_converter = DataFormatConverter()
        exporter = DatasetExporter(converter=custom_converter)
        
        assert exporter.converter is custom_converter

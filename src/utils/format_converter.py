"""
資料格式轉換工具
Data Format Converter Utility

支援多種資料格式之間的轉換：
- JSON/JSONL ↔ Arrow
- CSV ↔ JSONL
- 資料集匯入/匯出
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from ..constants.dataset import SUPPORTED_FORMATS, EXPORT_FORMATS
from ..models.dataset import ProcessedRecord
from ..models.quality import BatchQualityReport

logger = logging.getLogger(__name__)


class FormatConversionError(Exception):
    """格式轉換錯誤"""
    pass


class DataFormatConverter:
    """資料格式轉換器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".DataFormatConverter")
    
    def detect_format(self, file_path: Union[str, Path]) -> str:
        """
        檢測檔案格式
        
        Args:
            file_path: 檔案路徑
            
        Returns:
            檔案格式 ('jsonl', 'arrow', 'csv', 'parquet')
            
        Raises:
            FormatConversionError: 不支援的格式
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        for format_name, extensions in SUPPORTED_FORMATS.items():
            if suffix in extensions:
                return format_name
        
        raise FormatConversionError(f"Unsupported file format: {suffix}")
    
    def records_to_jsonl(self, records: List[ProcessedRecord], output_path: Union[str, Path]) -> None:
        """
        將處理後記錄轉換為 JSONL 格式
        
        Args:
            records: 處理後記錄列表
            output_path: 輸出檔案路徑
        """
        try:
            self.logger.info(f"Converting {len(records)} records to JSONL: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in records:
                    # 轉換為字典格式
                    record_dict = self.processed_record_to_dict(record)
                    # 寫入 JSONL
                    f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')
            
            self.logger.info(f"Successfully converted {len(records)} records to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to convert records to JSONL: {e}")
            raise FormatConversionError(f"JSONL conversion failed: {e}")
    
    def records_to_csv(self, records: List[ProcessedRecord], output_path: Union[str, Path]) -> None:
        """
        將處理後記錄轉換為 CSV 格式
        
        Args:
            records: 處理後記錄列表
            output_path: 輸出檔案路徑
        """
        try:
            import pandas as pd
            
            self.logger.info(f"Converting {len(records)} records to CSV: {output_path}")
            
            # 轉換為扁平化字典列表
            data_rows = []
            for record in records:
                row = {
                    'id': record.original_record.id,
                    'original_question': record.original_record.question,
                    'original_answer': record.original_record.answer,
                    'translated_question': record.translation_result.translated_question if record.translation_result else '',
                    'translated_answer': record.translation_result.translated_answer if record.translation_result else '',
                    'processing_status': record.processing_status.value,
                    'final_quality_score': record.final_quality_score,
                    'processing_time': record.processing_time,
                    'retry_count': record.retry_count,
                    'source_dataset': record.original_record.source_dataset
                }
                data_rows.append(row)
            
            # 創建 DataFrame 並保存
            df = pd.DataFrame(data_rows)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            self.logger.info(f"Successfully converted {len(records)} records to {output_path}")
            
        except ImportError:
            raise FormatConversionError("pandas not installed. Run: pip install pandas")
        except Exception as e:
            self.logger.error(f"Failed to convert records to CSV: {e}")
            raise FormatConversionError(f"CSV conversion failed: {e}")
    
    def records_to_arrow(self, records: List[ProcessedRecord], output_path: Union[str, Path]) -> None:
        """
        將處理後記錄轉換為 Arrow 格式
        
        Args:
            records: 處理後記錄列表
            output_path: 輸出檔案路徑
        """
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
            
            self.logger.info(f"Converting {len(records)} records to Arrow: {output_path}")
            
            # 轉換為字典列表
            data_dicts = [self.processed_record_to_dict(record) for record in records]
            
            # 創建 Arrow Table
            table = pa.Table.from_pylist(data_dicts)
            
            # 寫入 Arrow 檔案
            with pa.OSFile(str(output_path), 'wb') as sink:
                with ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)
            
            self.logger.info(f"Successfully converted {len(records)} records to {output_path}")
            
        except ImportError:
            raise FormatConversionError("pyarrow not installed. Run: pip install pyarrow")
        except Exception as e:
            self.logger.error(f"Failed to convert records to Arrow: {e}")
            raise FormatConversionError(f"Arrow conversion failed: {e}")
    
    def processed_record_to_dict(self, record: ProcessedRecord) -> Dict[str, Any]:
        """
        將處理後記錄轉換為字典
        
        Args:
            record: 處理後記錄
            
        Returns:
            字典表示
        """
        base_dict = {
            'id': record.original_record.id,
            'original': {
                'question': record.original_record.question,
                'answer': record.original_record.answer,
                'source_dataset': record.original_record.source_dataset,
                'metadata': record.original_record.metadata,
                'complexity_level': record.original_record.complexity_level.value if record.original_record.complexity_level else None
            },
            'processing_status': record.processing_status.value,
            'final_quality_score': record.final_quality_score,
            'processing_time': record.processing_time,
            'retry_count': record.retry_count
        }
        
        # 添加翻譯結果
        if record.translation_result:
            base_dict['translation'] = {
                'question': record.translation_result.translated_question,
                'answer': record.translation_result.translated_answer,
                'strategy': record.translation_result.translation_strategy,
                'terminology_notes': record.translation_result.terminology_notes,
                'timestamp': record.translation_result.timestamp
            }
        
        # 添加 QA 結果
        if record.original_qa_result:
            base_dict['original_qa'] = {
                'question': record.original_qa_result.input_question,
                'answer': record.original_qa_result.generated_answer,
                'execution_time': record.original_qa_result.execution_time,
                'reasoning_steps': record.original_qa_result.reasoning_steps,
                'confidence_score': record.original_qa_result.confidence_score
            }
        
        if record.translated_qa_result:
            base_dict['translated_qa'] = {
                'question': record.translated_qa_result.input_question,
                'answer': record.translated_qa_result.generated_answer,
                'execution_time': record.translated_qa_result.execution_time,
                'reasoning_steps': record.translated_qa_result.reasoning_steps,
                'confidence_score': record.translated_qa_result.confidence_score
            }
        
        return base_dict
    
    def export_records(self, 
                      records: List[ProcessedRecord], 
                      output_path: Union[str, Path],
                      format_type: str = 'jsonl') -> None:
        """
        匯出處理後記錄
        
        Args:
            records: 處理後記錄列表
            output_path: 輸出路徑
            format_type: 輸出格式 ('jsonl', 'csv', 'arrow')
        """
        if format_type not in EXPORT_FORMATS:
            raise FormatConversionError(f"Unsupported export format: {format_type}")
        
        # 確保輸出路徑有正確的副檔名
        path = Path(output_path)
        expected_ext = EXPORT_FORMATS[format_type]["extension"]
        if not path.suffix:
            path = path.with_suffix(expected_ext)
        elif path.suffix != expected_ext:
            self.logger.warning(f"Output path extension {path.suffix} doesn't match format {format_type}")
        
        # 根據格式類型調用相應的轉換方法
        if format_type == 'jsonl':
            self.records_to_jsonl(records, path)
        elif format_type == 'csv':
            self.records_to_csv(records, path)
        elif format_type == 'arrow':
            self.records_to_arrow(records, path)
        else:
            raise FormatConversionError(f"Conversion method not implemented for format: {format_type}")
    
    def export_quality_report(self, 
                             report: BatchQualityReport,
                             output_path: Union[str, Path]) -> None:
        """
        匯出品質報告
        
        Args:
            report: 批次品質報告
            output_path: 輸出路徑
        """
        try:
            self.logger.info(f"Exporting quality report to: {output_path}")
            
            report_dict = {
                'batch_id': report.batch_id,
                'summary': {
                    'total_records': report.total_records,
                    'processed_records': report.processed_records,
                    'passed_records': report.passed_records,
                    'failed_records': report.failed_records,
                    'retry_records': report.retry_records,
                    'success_rate': report.get_success_rate(),
                    'failure_rate': report.get_failure_rate()
                },
                'quality_statistics': {
                    'average_quality_score': report.average_quality_score,
                    'min_quality_score': report.min_quality_score,
                    'max_quality_score': report.max_quality_score
                },
                'processing_statistics': {
                    'total_processing_time': report.total_processing_time,
                    'average_processing_time': report.average_processing_time,
                    'total_retries': report.total_retries
                },
                'error_summary': {str(k): v for k, v in report.error_summary.items()},
                'timestamps': {
                    'batch_start_time': report.batch_start_time,
                    'batch_end_time': report.batch_end_time
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Quality report exported successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to export quality report: {e}")
            raise FormatConversionError(f"Quality report export failed: {e}")


class DatasetExporter:
    """資料集匯出器"""
    
    def __init__(self, converter: Optional[DataFormatConverter] = None):
        self.converter = converter or DataFormatConverter()
        self.logger = logging.getLogger(__name__ + ".DatasetExporter")
    
    def export_dataset(self,
                      records: List[ProcessedRecord],
                      output_dir: Union[str, Path],
                      dataset_name: str = "translated_dataset",
                      formats: List[str] = None) -> Dict[str, Path]:
        """
        匯出完整資料集
        
        Args:
            records: 處理後記錄列表
            output_dir: 輸出目錄
            dataset_name: 資料集名稱
            formats: 要匯出的格式列表
            
        Returns:
            格式與檔案路徑的映射
        """
        if formats is None:
            formats = ['jsonl', 'csv']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for format_type in formats:
            try:
                file_ext = EXPORT_FORMATS[format_type]["extension"]
                file_path = output_path / f"{dataset_name}{file_ext}"
                
                self.converter.export_records(records, file_path, format_type)
                exported_files[format_type] = file_path
                
            except Exception as e:
                self.logger.error(f"Failed to export {format_type}: {e}")
                continue
        
        # 如果沒有任何格式成功匯出，拋出錯誤
        if not exported_files:
            raise FormatConversionError("All export formats failed")
        
        self.logger.info(f"Dataset exported in {len(exported_files)} formats to {output_path}")
        return exported_files
    
    def export_successful_records_only(self,
                                     records: List[ProcessedRecord],
                                     output_dir: Union[str, Path],
                                     dataset_name: str = "successful_records") -> Dict[str, Path]:
        """
        僅匯出成功處理的記錄
        
        Args:
            records: 處理後記錄列表
            output_dir: 輸出目錄
            dataset_name: 資料集名稱
            
        Returns:
            格式與檔案路徑的映射
        """
        from ..models.dataset import ProcessingStatus
        
        successful_records = [
            record for record in records 
            if record.processing_status == ProcessingStatus.COMPLETED
        ]
        
        self.logger.info(f"Exporting {len(successful_records)} successful records out of {len(records)} total")
        
        return self.export_dataset(successful_records, output_dir, dataset_name)


# 預設轉換器實例
default_converter = DataFormatConverter()
default_exporter = DatasetExporter()


__all__ = [
    "FormatConversionError",
    "DataFormatConverter",
    "DatasetExporter", 
    "default_converter",
    "default_exporter",
]

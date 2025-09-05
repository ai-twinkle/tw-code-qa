"""
資料集管理核心模組
Dataset Manager Core Module

負責管理整個資料集處理管道：
1. 資料集載入與批次處理
2. 工作流程執行協調
3. 進度追蹤與錯誤恢復
4. 資料輸出與報告生成
"""

import logging
import time
import gc
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Union

from ..config.settings import get_config_for_environment, FEATURE_FLAGS
from ..constants.dataset import DEFAULT_BATCH_SIZE, PROGRESS_SAVE_INTERVAL, PROGRESS_LOG_INTERVAL
from ..models.dataset import OriginalRecord, ProcessedRecord, DatasetMetadata, ProcessingStatus
from ..models.quality import BatchQualityReport, QualityReport, ErrorRecord, ErrorType
from ..services.data_loader import DataLoaderFactory, DataLoadError
from ..workflow.graph import WorkflowManager
from ..workflow.state import create_initial_state, WorkflowState

logger = logging.getLogger(__name__)


class DatasetProcessingError(Exception):
    """資料集處理錯誤"""
    pass


class DatasetManager:
    """資料集管理器"""
    
    def __init__(self, 
                 output_dir: str = "output",
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 enable_checkpointing: bool = True):
        """
        初始化資料集管理器
        
        Args:
            output_dir: 輸出目錄
            batch_size: 批次大小
            enable_checkpointing: 是否啟用檢查點
        """
        self.logger = logging.getLogger(__name__ + ".DatasetManager")
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.enable_checkpointing = enable_checkpointing
        
        # 創建輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化工作流管理器
        self.workflow_manager = WorkflowManager(enable_checkpointing=enable_checkpointing)
        
        # 載入系統配置
        self.config = get_config_for_environment()
        
        # 處理統計
        self.total_processed = 0
        self.successful_records = 0
        self.failed_records = 0
        self.processing_start_time = 0.0
        
        self.logger.info(f"DatasetManager initialized with batch_size={batch_size}, output_dir={output_dir}")
    
    def load_dataset(self, dataset_path: str, dataset_type: str = "opencoder") -> Iterator[OriginalRecord]:
        """
        載入資料集
        
        Args:
            dataset_path: 資料集路徑
            dataset_type: 資料集類型
            
        Yields:
            原始記錄
        """
        try:
            self.logger.info(f"Loading dataset from {dataset_path}, type: {dataset_type}")
            
            # 創建資料載入器
            data_loader = DataLoaderFactory.create_loader(dataset_type)
            
            # 載入資料集
            for record in data_loader.load_dataset(dataset_path):
                yield record
                
        except DataLoadError as e:
            self.logger.error(f"Data loading failed: {e}")
            raise DatasetProcessingError(f"Failed to load dataset: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during data loading: {e}")
            raise DatasetProcessingError(f"Unexpected error: {e}")
    
    def process_record(self, record: OriginalRecord) -> ProcessedRecord:
        """
        處理單一記錄
        
        Args:
            record: 原始記錄
            
        Returns:
            處理後記錄
        """
        start_time = time.time()
        
        try:
            # 創建初始工作流狀態
            initial_state = create_initial_state(record)
            
            # 生成配置（用於檢查點）
            config = {"configurable": {"thread_id": record.id}} if self.enable_checkpointing else None
            
            # 執行工作流
            final_state = self.workflow_manager.process_record(initial_state, config)
            
            # 計算處理時間
            processing_time = time.time() - start_time
            
            # 提取結果
            translation_result = final_state.get("translation_result")
            original_qa_result = final_state.get("original_qa_result")
            translated_qa_result = final_state.get("translated_qa_result")
            quality_assessment = final_state.get("quality_assessment")
            processing_status = final_state.get("processing_status", ProcessingStatus.FAILED)
            retry_count = final_state.get("retry_count", 0)
            
            # 計算最終品質分數
            final_quality_score = (quality_assessment.overall_quality_score 
                                 if quality_assessment else 0.0)
            
            # 創建處理後記錄
            processed_record = ProcessedRecord(
                original_record=record,
                translation_result=translation_result,
                original_qa_result=original_qa_result,
                translated_qa_result=translated_qa_result,
                processing_status=processing_status,
                final_quality_score=final_quality_score,
                processing_time=processing_time,
                retry_count=retry_count
            )
            
            # 更新統計
            if processing_status == ProcessingStatus.COMPLETED:
                self.successful_records += 1
            else:
                self.failed_records += 1
            
            self.total_processed += 1
            
            return processed_record
            
        except Exception as e:
            self.logger.error(f"Failed to process record {record.id}: {e}")
            
            # 創建失敗記錄
            processing_time = time.time() - start_time
            self.failed_records += 1
            self.total_processed += 1
            
            return ProcessedRecord(
                original_record=record,
                translation_result=None,
                original_qa_result=None,
                translated_qa_result=None,
                processing_status=ProcessingStatus.FAILED,
                final_quality_score=0.0,
                processing_time=processing_time,
                retry_count=0
            )
    
    def process_batch(self, records: List[OriginalRecord]) -> List[ProcessedRecord]:
        """
        處理記錄批次
        
        Args:
            records: 記錄列表
            
        Returns:
            處理後記錄列表
        """
        batch_id = f"batch_{int(time.time())}"
        self.logger.info(f"Processing batch {batch_id} with {len(records)} records")
        
        processed_records = []
        
        for i, record in enumerate(records):
            try:
                self.logger.debug(f"Processing record {i+1}/{len(records)}: {record.id}")
                processed_record = self.process_record(record)
                processed_records.append(processed_record)
                
                # 記錄進度
                if (i + 1) % PROGRESS_LOG_INTERVAL == 0:
                    self.logger.info(f"Batch {batch_id} progress: {i+1}/{len(records)} records processed")
                
            except Exception as e:
                self.logger.error(f"Critical error processing record {record.id}: {e}")
                continue
        
        # 執行垃圾回收
        if FEATURE_FLAGS.get("enable_memory_optimization", True):
            gc.collect()
        
        self.logger.info(f"Batch {batch_id} completed: {len(processed_records)} records processed")
        return processed_records
    
    def process_dataset(self, 
                       dataset_path: Union[str, Iterator[OriginalRecord]],
                       dataset_type: str = "opencoder",
                       max_records: Optional[int] = None) -> BatchQualityReport:
        """
        處理完整資料集
        
        Args:
            dataset_path: 資料集路徑或原始記錄的迭代器
            dataset_type: 資料集類型
            max_records: 最大處理記錄數（用於測試）
            
        Returns:
            批次品質報告
        """
        self.processing_start_time = time.time()
        self.logger.info(f"Starting dataset processing: {dataset_path}")
        
        try:
            # 重置統計
            self.total_processed = 0
            self.successful_records = 0
            self.failed_records = 0
            
            # 載入資料集 - 判斷是路徑還是迭代器
            if isinstance(dataset_path, str):
                dataset_iterator = self.load_dataset(dataset_path, dataset_type)
            else:
                # 假設是迭代器
                dataset_iterator = dataset_path
            
            # 批次處理
            current_batch = []
            all_processed_records = []
            
            for i, record in enumerate(dataset_iterator):
                # 檢查最大記錄數限制
                if max_records and i >= max_records:
                    self.logger.info(f"Reached max_records limit: {max_records}")
                    break
                
                current_batch.append(record)
                
                # 當批次滿了或是最後一筆記錄時處理
                if len(current_batch) >= self.batch_size:
                    processed_records = self.process_batch(current_batch)
                    all_processed_records.extend(processed_records)
                    
                    # 保存中間結果
                    if FEATURE_FLAGS.get("enable_progress_save", True):
                        self._save_intermediate_results(all_processed_records)
                    
                    current_batch = []
            
            # 處理剩餘記錄
            if current_batch:
                processed_records = self.process_batch(current_batch)
                all_processed_records.extend(processed_records)
            
            # 生成最終報告
            batch_report = self._generate_batch_report(all_processed_records)
            
            # 保存最終結果
            self._save_final_results(all_processed_records, batch_report)
            
            total_time = time.time() - self.processing_start_time
            self.logger.info(f"Dataset processing completed in {total_time:.2f}s")
            self.logger.info(f"Total: {self.total_processed}, Success: {self.successful_records}, Failed: {self.failed_records}")
            
            return batch_report
            
        except Exception as e:
            self.logger.error(f"Dataset processing failed: {e}")
            raise DatasetProcessingError(f"Dataset processing failed: {e}")
    
    def _save_intermediate_results(self, processed_records: List[ProcessedRecord]):
        """保存中間結果"""
        try:
            output_file = self.output_dir / f"intermediate_results_{int(time.time())}.jsonl"
            
            # 簡化的保存邏輯
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in processed_records[-self.batch_size:]:  # 只保存最新的批次
                    # 這裡應該實現 JSON 序列化
                    f.write(f"{{'record_id': '{record.original_record.id}', 'status': '{record.processing_status.value}'}}\n")
            
            self.logger.debug(f"Intermediate results saved to {output_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save intermediate results: {e}")
    
    def _save_final_results(self, processed_records: List[ProcessedRecord], batch_report: BatchQualityReport):
        """保存最終結果"""
        try:
            # 保存處理後的記錄
            records_file = self.output_dir / "processed_records.jsonl"
            with open(records_file, 'w', encoding='utf-8') as f:
                for record in processed_records:
                    # 這裡應該實現完整的 JSON 序列化
                    f.write(f"{{'record_id': '{record.original_record.id}', 'status': '{record.processing_status.value}'}}\n")
            
            # 保存品質報告
            report_file = self.output_dir / "quality_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                # 這裡應該實現品質報告的 JSON 序列化
                f.write(f"{{'total_records': {batch_report.total_records}, 'success_rate': {batch_report.get_success_rate()}}}\n")
            
            self.logger.info(f"Final results saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save final results: {e}")
    
    def _generate_batch_report(self, processed_records: List[ProcessedRecord]) -> BatchQualityReport:
        """生成批次品質報告"""
        try:
            batch_id = f"batch_{int(self.processing_start_time)}"
            
            # 計算統計
            total_records = len(processed_records)
            passed_records = len([r for r in processed_records if r.processing_status == ProcessingStatus.COMPLETED])
            failed_records = len([r for r in processed_records if r.processing_status == ProcessingStatus.FAILED])
            retry_records = sum(r.retry_count for r in processed_records)
            
            # 計算品質分數統計
            quality_scores = [r.final_quality_score for r in processed_records if r.final_quality_score > 0]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            min_quality = min(quality_scores) if quality_scores else 0.0
            max_quality = max(quality_scores) if quality_scores else 0.0
            
            # 計算處理時間統計
            processing_times = [r.processing_time for r in processed_records]
            total_processing_time = sum(processing_times)
            avg_processing_time = total_processing_time / len(processing_times) if processing_times else 0.0
            
            # 創建報告
            batch_report = BatchQualityReport(
                batch_id=batch_id,
                total_records=total_records,
                processed_records=total_records,
                passed_records=passed_records,
                failed_records=failed_records,
                retry_records=retry_records,
                average_quality_score=avg_quality,
                min_quality_score=min_quality,
                max_quality_score=max_quality,
                total_processing_time=total_processing_time,
                average_processing_time=avg_processing_time,
                total_retries=retry_records,
                batch_start_time=self.processing_start_time,
                batch_end_time=time.time()
            )
            
            # 計算統計
            batch_report.calculate_statistics()
            
            return batch_report
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch report: {e}")
            # 返回基本報告
            return BatchQualityReport(
                batch_id="error_batch",
                total_records=len(processed_records),
                processed_records=len(processed_records),
                passed_records=0,
                failed_records=len(processed_records),
                retry_records=0,
                average_quality_score=0.0,
                min_quality_score=0.0,
                max_quality_score=0.0,
                total_processing_time=0.0,
                average_processing_time=0.0,
                total_retries=0
            )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """獲取處理統計資訊"""
        current_time = time.time()
        elapsed_time = current_time - self.processing_start_time if self.processing_start_time > 0 else 0
        
        return {
            "total_processed": self.total_processed,
            "successful_records": self.successful_records,
            "failed_records": self.failed_records,
            "success_rate": self.successful_records / max(self.total_processed, 1),
            "processing_time": elapsed_time,
            "records_per_minute": (self.total_processed / (elapsed_time / 60)) if elapsed_time > 0 else 0
        }
    
    def save_results(self, processed_records: List[ProcessedRecord], output_dir: str, dataset_name: str) -> None:
        """
        保存處理結果到指定目錄
        
        Args:
            processed_records: 處理後的記錄列表
            output_dir: 輸出目錄
            dataset_name: 資料集名稱
            
        Raises:
            DatasetProcessingError: 保存失敗時拋出
        """
        try:
            from ..utils.format_converter import DatasetExporter
            
            # 創建輸出目錄
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 創建導出器並保存
            exporter = DatasetExporter()
            exporter.export_dataset(
                records=processed_records,
                output_dir=output_path,
                dataset_name=dataset_name,
                formats=["jsonl"]  # 預設格式
            )
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise DatasetProcessingError(f"Failed to save results: {e}")


__all__ = [
    "DatasetManager",
    "DatasetProcessingError",
]

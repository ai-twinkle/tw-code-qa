"""
資料集管理核心模組 - 重寫版本
Dataset Manager Core Module - Rewritten Version

核心邏輯：
1. 執行一筆保存一筆
2. Resume 檢查已跑結果，找失敗和缺失記錄重跑
3. 完成後合併所有成功記錄
"""

import gc
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..config.settings import get_config_for_environment, FEATURE_FLAGS
from ..constants.dataset import PROGRESS_LOG_INTERVAL
from ..models.dataset import OriginalRecord, ProcessedRecord, ProcessingStatus
from ..models.quality import BatchQualityReport
from ..services.data_loader import DataLoaderFactory, DataLoadError
from ..utils.format_converter import DataFormatConverter
from ..utils.recovery import RecoveryManager
from ..workflow.graph import WorkflowManager
from ..workflow.state import create_initial_state

logger = logging.getLogger(__name__)


class DatasetProcessingError(Exception):
    """資料集處理錯誤"""
    pass


class DatasetManager:
    """簡潔的資料集管理器"""
    
    def __init__(self, output_dir: str = "output", enable_checkpointing: bool = True):
        """
        初始化資料集管理器
        
        Args:
            output_dir: 輸出目錄
            enable_checkpointing: 是否啟用檢查點
        """
        self.logger = logging.getLogger(__name__ + ".DatasetManager")
        self.output_dir = Path(output_dir)
        self.enable_checkpointing = enable_checkpointing
        
        # 創建輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化核心組件
        self.workflow_manager = WorkflowManager(enable_checkpointing=enable_checkpointing)
        self.format_converter = DataFormatConverter()
        self.recovery_manager = RecoveryManager(str(self.output_dir))
        
        # 載入系統配置
        self.config = get_config_for_environment()
        
        self.logger.info(f"DatasetManager initialized: output_dir={output_dir}")
    
    def load_dataset(self, dataset_path: str, dataset_type: str = "opencoder") -> List[OriginalRecord]:
        """
        載入完整資料集
        
        Args:
            dataset_path: 資料集路徑
            dataset_type: 資料集類型
            
        Returns:
            原始記錄列表
        """
        try:
            self.logger.info(f"Loading dataset from {dataset_path}, type: {dataset_type}")
            
            data_loader = DataLoaderFactory.create_loader(dataset_type)
            records = list(data_loader.load_dataset(dataset_path))
            
            self.logger.info(f"Loaded {len(records)} records")
            return records
                
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
            
            return processed_record
            
        except Exception as e:
            self.logger.error(f"Failed to process record {record.id}: {e}")
            
            # 創建失敗記錄
            processing_time = time.time() - start_time
            
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
    
    def run(self, 
            dataset_path: str,
            dataset_type: str = "opencoder",
            max_records: Optional[int] = None) -> BatchQualityReport:
        """
        運行資料集處理 - 執行一筆保存一筆
        
        Args:
            dataset_path: 資料集路徑
            dataset_type: 資料集類型
            max_records: 最大處理記錄數（用於測試）
            
        Returns:
            批次品質報告
        """
        start_time = time.time()
        self.logger.info(f"Starting dataset processing: {dataset_path}")
        
        try:
            # 載入所有記錄
            all_records = self.load_dataset(dataset_path, dataset_type)
            
            if max_records:
                all_records = all_records[:max_records]
                self.logger.info(f"Limited to {max_records} records for testing")
            
            # 逐筆處理並立即保存
            processed_count = 0
            for i, record in enumerate(all_records):
                self.logger.info(f"Processing record {i+1}/{len(all_records)}: {record.id}")
                
                # 處理記錄
                processed_record = self.process_record(record)
                
                # 立即保存
                success = self.recovery_manager.save_record(processed_record)
                if success:
                    self.logger.debug(f"Record {record.id} saved immediately")
                else:
                    self.logger.warning(f"Failed to save record {record.id}")
                
                processed_count += 1
                
                # 定期記錄進度
                if processed_count % PROGRESS_LOG_INTERVAL == 0:
                    self.logger.info(f"Progress: {processed_count}/{len(all_records)} records processed")
                
                # 內存優化
                if FEATURE_FLAGS.get("enable_memory_optimization", True) and processed_count % 50 == 0:
                    gc.collect()
            
            # 生成報告
            report = self._create_batch_report(all_records, start_time)
            self._save_final_results(report)
            
            total_time = time.time() - start_time
            self.logger.info(f"Dataset processing completed in {total_time:.2f}s")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Dataset processing failed: {e}")
            raise DatasetProcessingError(f"Dataset processing failed: {e}")
    
    def resume(self, 
               dataset_path: str,
               dataset_type: str = "opencoder",
               max_records: Optional[int] = None) -> BatchQualityReport:
        """
        恢復處理 - 找到失敗和缺失記錄重跑，然後合併
        
        Args:
            dataset_path: 資料集路徑
            dataset_type: 資料集類型
            max_records: 最大處理記錄數（用於測試）
            
        Returns:
            批次品質報告
        """
        start_time = time.time()
        self.logger.info("Starting resume processing...")
        
        try:
            # 載入所有記錄
            all_records = self.load_dataset(dataset_path, dataset_type)
            
            if max_records:
                all_records = all_records[:max_records]
            
            all_record_ids = [record.id for record in all_records]
            
            # 分析當前狀態
            self.recovery_manager.print_status(all_record_ids)
            analysis = self.recovery_manager.find_missing_and_failed(all_record_ids)
            
            need_processing_ids = analysis["need_processing"]
            self.logger.info(f"Found {len(need_processing_ids)} records that need processing")
            
            if not need_processing_ids:
                self.logger.info("All records already processed successfully!")
                # 載入現有成功記錄生成報告
                successful_records = self.recovery_manager.load_successful_records()
                return self._create_batch_report_from_records(successful_records, start_time)
            
            # 找到需要處理的記錄
            records_to_process = [
                record for record in all_records 
                if record.id in need_processing_ids
            ]
            
            self.logger.info(f"Processing {len(records_to_process)} records...")
            
            # 處理需要的記錄
            processed_count = 0
            for i, record in enumerate(records_to_process):
                self.logger.info(f"Processing record {i+1}/{len(records_to_process)}: {record.id}")
                
                # 處理記錄
                processed_record = self.process_record(record)
                
                # 立即保存
                success = self.recovery_manager.save_record(processed_record)
                if success:
                    self.logger.debug(f"Record {record.id} saved immediately")
                else:
                    self.logger.warning(f"Failed to save record {record.id}")
                
                processed_count += 1
                
                # 定期記錄進度
                if processed_count % PROGRESS_LOG_INTERVAL == 0:
                    self.logger.info(f"Resume progress: {processed_count}/{len(records_to_process)} records processed")
                
                # 內存優化
                if FEATURE_FLAGS.get("enable_memory_optimization", True) and processed_count % 20 == 0:
                    gc.collect()
            
            # 載入所有成功記錄並生成最終報告
            self.logger.info("Loading all successful records for final report...")
            all_successful_records = self.recovery_manager.load_successful_records()
            
            report = self._create_batch_report_from_records(all_successful_records, start_time)
            self._save_final_results(report)
            
            total_time = time.time() - start_time
            self.logger.info(f"Resume processing completed in {total_time:.2f}s")
            self.logger.info(f"Final successful records: {len(all_successful_records)}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Resume processing failed: {e}")
            raise DatasetProcessingError(f"Resume processing failed: {e}")
    
    def _create_batch_report(self, all_records: List[OriginalRecord], start_time: float) -> BatchQualityReport:
        """
        從所有記錄創建批次報告
        
        Args:
            all_records: 所有原始記錄
            start_time: 開始時間
            
        Returns:
            批次品質報告
        """
        # 獲取處理狀態
        status = self.recovery_manager.get_processed_status()
        
        total_records = len(all_records)
        passed_records = len(status["successful"])
        failed_records = len(status["failed"])
        
        # 載入成功記錄計算品質分數
        successful_records = self.recovery_manager.load_successful_records()
        quality_scores = [r.final_quality_score for r in successful_records if r.final_quality_score > 0]
        processing_times = [r.processing_time for r in successful_records]
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        min_quality = min(quality_scores) if quality_scores else 0.0
        max_quality = max(quality_scores) if quality_scores else 0.0
        
        total_processing_time = sum(processing_times) if processing_times else 0.0
        avg_processing_time = total_processing_time / len(processing_times) if processing_times else 0.0
        
        total_retries = sum(r.retry_count for r in successful_records)
        
        return BatchQualityReport(
            batch_id=f"batch_{int(start_time)}",
            total_records=total_records,
            processed_records=passed_records + failed_records,
            passed_records=passed_records,
            failed_records=failed_records,
            retry_records=total_retries,
            average_quality_score=avg_quality,
            min_quality_score=min_quality,
            max_quality_score=max_quality,
            total_processing_time=total_processing_time,
            average_processing_time=avg_processing_time,
            total_retries=total_retries,
            batch_start_time=start_time,
            batch_end_time=time.time()
        )

    @staticmethod
    def _create_batch_report_from_records(records: List[ProcessedRecord], start_time: float) -> BatchQualityReport:
        """
        從處理後記錄創建批次報告
        
        Args:
            records: 處理後記錄列表
            start_time: 開始時間
            
        Returns:
            批次品質報告
        """
        total_records = len(records)
        passed_records = len([r for r in records if r.processing_status == ProcessingStatus.COMPLETED])
        failed_records = total_records - passed_records
        
        quality_scores = [r.final_quality_score for r in records if r.final_quality_score > 0]
        processing_times = [r.processing_time for r in records]
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        min_quality = min(quality_scores) if quality_scores else 0.0
        max_quality = max(quality_scores) if quality_scores else 0.0
        
        total_processing_time = sum(processing_times) if processing_times else 0.0
        avg_processing_time = total_processing_time / len(processing_times) if processing_times else 0.0
        
        total_retries = sum(r.retry_count for r in records)
        
        return BatchQualityReport(
            batch_id=f"batch_{int(start_time)}",
            total_records=total_records,
            processed_records=total_records,
            passed_records=passed_records,
            failed_records=failed_records,
            retry_records=total_retries,
            average_quality_score=avg_quality,
            min_quality_score=min_quality,
            max_quality_score=max_quality,
            total_processing_time=total_processing_time,
            average_processing_time=avg_processing_time,
            total_retries=total_retries,
            batch_start_time=start_time,
            batch_end_time=time.time()
        )
    
    def _save_final_results(self, batch_report: BatchQualityReport):
        """保存最終結果"""
        try:
            # 保存品質報告
            report_file = self.output_dir / "quality_report.json"
            self.format_converter.export_quality_report(batch_report, report_file)
            
            self.logger.info(f"Final results saved to {self.output_dir}")
            self.logger.info(f"Quality report: {report_file}")
            self.logger.info(f"Processed records: {self.recovery_manager.results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save final results: {e}")
    
    def get_status(self, dataset_path: Optional[str] = None, dataset_type: str = "opencoder") -> Dict[str, Any]:
        """
        獲取當前處理狀態
        
        Args:
            dataset_path: 資料集路徑（可選，用於完整分析）
            dataset_type: 資料集類型
            
        Returns:
            狀態字典
        """
        if dataset_path:
            all_records = self.load_dataset(dataset_path, dataset_type)
            all_record_ids = [record.id for record in all_records]
            return self.recovery_manager.find_missing_and_failed(all_record_ids)
        else:
            return self.recovery_manager.get_processed_status()
    
    def print_status(self, dataset_path: Optional[str] = None, dataset_type: str = "opencoder") -> None:
        """
        打印當前處理狀態
        
        Args:
            dataset_path: 資料集路徑（可選，用於完整分析）
            dataset_type: 資料集類型
        """
        if dataset_path:
            all_records = self.load_dataset(dataset_path, dataset_type)
            all_record_ids = [record.id for record in all_records]
            self.recovery_manager.print_status(all_record_ids)
        else:
            self.recovery_manager.print_status()


__all__ = [
    "DatasetManager",
    "DatasetProcessingError",
]

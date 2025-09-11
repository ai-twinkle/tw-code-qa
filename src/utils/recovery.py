"""
簡潔的恢復管理模組
Simple Recovery Manager Module

實現：
1. 分析已處理記錄狀態
2. 識別失敗和缺失記錄
3. 支持記錄級別的即時保存
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from ..models.dataset import ProcessedRecord, ProcessingStatus
from .format_converter import DataFormatConverter

logger = logging.getLogger(__name__)


class RecoveryManager:
    """簡潔的恢復管理器"""
    
    def __init__(self, output_dir: str):
        """
        初始化恢復管理器
        
        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.results_file = self.output_dir / "processed_records.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用現有的格式轉換器
        self.format_converter = DataFormatConverter()
    
    def save_record(self, record: ProcessedRecord) -> bool:
        """
        保存單筆記錄（會替換同ID的舊記錄）
        
        Args:
            record: 處理後的記錄
            
        Returns:
            是否保存成功
        """
        try:
            # 使用格式轉換器將記錄轉換為字典
            record_dict = self.format_converter.processed_record_to_dict(record)
            record_id = record.original_record.id
            
            # 讀取現有記錄
            existing_records = []
            if self.results_file.exists():
                with open(self.results_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            existing_record = json.loads(line.strip())
                            # 如果不是同一個ID，保留現有記錄
                            if existing_record.get("id") != record_id:
                                existing_records.append(existing_record)
                        except json.JSONDecodeError:
                            continue
            
            # 添加新記錄
            existing_records.append(record_dict)
            
            # 重寫文件
            with open(self.results_file, "w", encoding="utf-8") as f:
                for record_entry in existing_records:
                    f.write(json.dumps(record_entry, ensure_ascii=False) + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"保存記錄失敗 {record.original_record.id}: {e}")
            return False
    
    def get_processed_status(self) -> Dict[str, Any]:
        """
        獲取已處理記錄的狀態
        
        Returns:
            包含成功、失敗記錄 ID 的字典
        """
        successful_ids = set()
        failed_ids = set()
        
        if not self.results_file.exists():
            return {
                "successful": successful_ids,
                "failed": failed_ids,
                "total": 0
            }
        
        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        record_id = record.get("id")
                        status = record.get("processing_status")
                        
                        if record_id and status:
                            if status == ProcessingStatus.COMPLETED.value:
                                # 進一步檢查記錄完整性
                                if self._is_record_complete(record):
                                    successful_ids.add(record_id)
                                    # 如果之前失敗過，現在成功了，從失敗列表移除
                                    failed_ids.discard(record_id)
                                else:
                                    # 記錄不完整，標記為失敗
                                    failed_ids.add(record_id)
                                    successful_ids.discard(record_id)
                            else:
                                # 只有在不是成功狀態時才加入失敗列表
                                if record_id not in successful_ids:
                                    failed_ids.add(record_id)
                                    
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"讀取處理狀態失敗: {e}")
        
        return {
            "successful": successful_ids,
            "failed": failed_ids,
            "total": len(successful_ids) + len(failed_ids)
        }
    
    def _is_record_complete(self, record: Dict[str, Any]) -> bool:
        """
        檢查記錄是否完整
        
        Args:
            record: 記錄字典
            
        Returns:
            記錄是否完整
        """
        try:
            # 檢查基本必要欄位
            basic_required_fields = ['id', 'original', 'processing_status', 'final_quality_score', 'processing_time', 'retry_count']
            for field in basic_required_fields:
                if field not in record:
                    logger.warning(f"Record {record.get('id', 'unknown')} missing required field: {field}")
                    return False
            
            # 檢查 original 欄位完整性
            original = record.get('original', {})
            original_required = ['question', 'answer', 'source_dataset', 'metadata', 'complexity_level']
            for field in original_required:
                if field not in original:
                    logger.warning(f"Record {record['id']} missing original.{field}")
                    return False
                # 檢查非空
                if field in ['question', 'answer'] and not original[field]:
                    logger.warning(f"Record {record['id']} has empty original.{field}")
                    return False
            
            # 如果狀態是 completed，檢查更詳細的完整性
            if record.get('processing_status') == ProcessingStatus.COMPLETED.value:
                
                # 檢查翻譯結果完整性
                if 'translation' not in record:
                    logger.warning(f"Record {record['id']} missing translation")
                    return False
                
                translation = record['translation']
                translation_required = ['question', 'answer', 'strategy', 'terminology_notes', 'timestamp']
                for field in translation_required:
                    if field not in translation:
                        logger.warning(f"Record {record['id']} missing translation.{field}")
                        return False
                    # 檢查關鍵欄位非空
                    if field in ['question', 'answer'] and not translation[field]:
                        logger.warning(f"Record {record['id']} has empty translation.{field}")
                        return False
                
                # 檢查 original_qa 結果完整性
                if 'original_qa' not in record:
                    logger.warning(f"Record {record['id']} missing original_qa")
                    return False
                
                original_qa = record['original_qa']
                original_qa_required = ['question', 'answer', 'execution_time', 'reasoning_steps', 'confidence_score']
                for field in original_qa_required:
                    if field not in original_qa:
                        logger.warning(f"Record {record['id']} missing original_qa.{field}")
                        return False
                    # 檢查關鍵欄位非空
                    if field in ['question', 'answer'] and not original_qa[field]:
                        logger.warning(f"Record {record['id']} has empty original_qa.{field}")
                        return False
                
                # 檢查 translated_qa 結果完整性
                if 'translated_qa' not in record:
                    logger.warning(f"Record {record['id']} missing translated_qa")
                    return False
                
                translated_qa = record['translated_qa']
                translated_qa_required = ['question', 'answer', 'execution_time', 'reasoning_steps', 'confidence_score']
                for field in translated_qa_required:
                    if field not in translated_qa:
                        logger.warning(f"Record {record['id']} missing translated_qa.{field}")
                        return False
                    # 檢查關鍵欄位非空
                    if field in ['question', 'answer'] and not translated_qa[field]:
                        logger.warning(f"Record {record['id']} has empty translated_qa.{field}")
                        return False
                
                # 檢查數值欄位的合理性
                if not isinstance(record.get('final_quality_score'), (int, float)) or record['final_quality_score'] < 0:
                    logger.warning(f"Record {record['id']} has invalid final_quality_score")
                    return False
                
                if not isinstance(record.get('processing_time'), (int, float)) or record['processing_time'] < 0:
                    logger.warning(f"Record {record['id']} has invalid processing_time")
                    return False
                
                if not isinstance(record.get('retry_count'), int) or record['retry_count'] < 0:
                    logger.warning(f"Record {record['id']} has invalid retry_count")
                    return False
                
                # 檢查執行時間欄位
                for qa_type in ['original_qa', 'translated_qa']:
                    qa_data = record[qa_type]
                    if not isinstance(qa_data.get('execution_time'), (int, float)) or qa_data['execution_time'] < 0:
                        logger.warning(f"Record {record['id']} has invalid {qa_type}.execution_time")
                        return False
                    
                    if not isinstance(qa_data.get('confidence_score'), (int, float)) or not (0 <= qa_data['confidence_score'] <= 1):
                        logger.warning(f"Record {record['id']} has invalid {qa_type}.confidence_score")
                        return False
                
                # 檢查時間戳 (支援數字和字串格式)
                timestamp = translation.get('timestamp')
                if not timestamp or (not isinstance(timestamp, (int, float, str)) or 
                                   (isinstance(timestamp, str) and timestamp.strip() == "")):
                    logger.warning(f"Record {record['id']} has invalid translation.timestamp")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking record completeness for {record.get('id', 'unknown')}: {e}")
            return False
    
    def find_missing_and_failed(self, all_record_ids: List[str]) -> Dict[str, Set[str]]:
        """
        找到缺失和失敗的記錄 ID
        
        Args:
            all_record_ids: 所有記錄的 ID 列表
            
        Returns:
            包含 missing 和 failed 記錄 ID 的字典
        """
        status = self.get_processed_status()
        all_ids = set(all_record_ids)
        processed_ids = status["successful"] | status["failed"]
        
        missing_ids = all_ids - processed_ids
        failed_ids = status["failed"]
        
        return {
            "missing": missing_ids,
            "failed": failed_ids,
            "need_processing": missing_ids | failed_ids
        }
    
    def print_status(self, all_record_ids: Optional[List[str]] = None) -> None:
        """
        打印當前狀態
        
        Args:
            all_record_ids: 所有記錄 ID（可選，用於計算缺失記錄）
        """
        status = self.get_processed_status()
        print("=" * 50)
        print("處理狀態報告")
        print("=" * 50)
        print(f"成功處理: {len(status['successful'])} 筆")
        print(f"處理失敗: {len(status['failed'])} 筆")
        print(f"總已處理: {status['total']} 筆")
        
        if all_record_ids:
            analysis = self.find_missing_and_failed(all_record_ids)
            print(f"尚未處理: {len(analysis['missing'])} 筆")
            print(f"需要重跑: {len(analysis['need_processing'])} 筆")
            print(f"總記錄數: {len(all_record_ids)} 筆")
            
            if status['total'] > 0:
                success_rate = len(status['successful']) / status['total'] * 100
                print(f"成功率: {success_rate:.1f}%")
        
        print("=" * 50)
    
    def load_successful_records(self) -> List[ProcessedRecord]:
        """
        載入所有成功處理的記錄
        
        Returns:
            成功記錄列表
        """
        successful_records = []
        
        if not self.results_file.exists():
            return successful_records
        
        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record_dict = json.loads(line.strip())
                        if record_dict.get("processing_status") == ProcessingStatus.COMPLETED.value:
                            # 重建 ProcessedRecord 對象（根據 DataFormatConverter 的格式）
                            from ..models.dataset import OriginalRecord, TranslationResult, QAExecutionResult, ComplexityLevel, Language
                            
                            # 重建 OriginalRecord（根據 DataFormatConverter 的格式）
                            original_data = record_dict["original"]
                            original_record = OriginalRecord(
                                id=record_dict["id"],
                                question=original_data["question"],
                                answer=original_data["answer"],
                                source_dataset=original_data["source_dataset"],
                                metadata=original_data["metadata"],
                                complexity_level=ComplexityLevel(original_data["complexity_level"]) if original_data.get("complexity_level") else None
                            )
                            
                            translation_result = None
                            if record_dict.get("translation"):
                                trans_data = record_dict["translation"]
                                translation_result = TranslationResult(
                                    original_record_id=record_dict["id"],
                                    translated_question=trans_data["question"],
                                    translated_answer=trans_data["answer"],
                                    translation_strategy=trans_data["strategy"],
                                    terminology_notes=trans_data["terminology_notes"],
                                    timestamp=trans_data["timestamp"]
                                )
                            
                            original_qa_result = None
                            if record_dict.get("original_qa"):
                                qa_data = record_dict["original_qa"]
                                original_qa_result = QAExecutionResult(
                                    record_id=record_dict["id"],
                                    language=Language.ENGLISH,
                                    input_question=qa_data["question"],
                                    generated_answer=qa_data["answer"],
                                    execution_time=qa_data["execution_time"],
                                    reasoning_steps=qa_data["reasoning_steps"],
                                    confidence_score=qa_data["confidence_score"],
                                    timestamp=""
                                )
                            
                            translated_qa_result = None
                            if record_dict.get("translated_qa"):
                                qa_data = record_dict["translated_qa"]
                                translated_qa_result = QAExecutionResult(
                                    record_id=record_dict["id"],
                                    language=Language.TRADITIONAL_CHINESE,
                                    input_question=qa_data["question"],
                                    generated_answer=qa_data["answer"],
                                    execution_time=qa_data["execution_time"],
                                    reasoning_steps=qa_data["reasoning_steps"],
                                    confidence_score=qa_data["confidence_score"],
                                    timestamp=""
                                )
                            
                            processed_record = ProcessedRecord(
                                original_record=original_record,
                                translation_result=translation_result,
                                original_qa_result=original_qa_result,
                                translated_qa_result=translated_qa_result,
                                processing_status=ProcessingStatus(record_dict["processing_status"]),
                                final_quality_score=record_dict["final_quality_score"],
                                processing_time=record_dict["processing_time"],
                                retry_count=record_dict["retry_count"]
                            )
                            
                            successful_records.append(processed_record)
                            
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"跳過無效記錄: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"載入成功記錄失敗: {e}")
        
        return successful_records


__all__ = ["RecoveryManager"]

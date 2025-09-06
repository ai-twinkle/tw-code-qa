"""
繁體中文程式碼問答資料集轉換系統 - 主程式入口
Traditional Chinese Code-QA Dataset Conversion System - Main Entry Point

Multi-Agent 架構的資料集轉換系統主程式
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.config.logging_config import setup_logging
from src.config.settings import set_environment
from src.core.dataset_manager import DatasetManager
from src.utils.format_converter import DatasetExporter


def setup_argument_parser() -> argparse.ArgumentParser:
    """設定命令列參數解析器"""
    parser = argparse.ArgumentParser(
        description='繁體中文程式碼問答資料集轉換系統 (Multi-Agent)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  # 處理 OpenCoder 教育指導資料集
  python main.py --dataset-path data/opencoder_dataset_educational_instruct --dataset-type opencoder
  
  # 指定輸出目錄和批次大小
  python main.py --dataset-path data/opencoder_dataset_evol_instruct --output-dir output/evol --batch-size 50
  
  # 測試模式 (只處理前 10 筆記錄)
  python main.py --dataset-path data/sample --max-records 10 --environment development
  
  # 生產模式
  python main.py --dataset-path data/opencoder_dataset_package_instruct --environment production --batch-size 200
        """
    )
    
    # 必須參數
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='資料集路徑 (必須)'
    )
    
    # 可選參數
    parser.add_argument(
        '--dataset-type',
        type=str,
        default='opencoder',
        choices=['opencoder'],
        help='資料集類型 (預設: opencoder)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='輸出目錄 (預設: output)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='批次大小 (預設: 100)'
    )
    
    parser.add_argument(
        '--max-records',
        type=int,
        help='最大處理記錄數 (用於測試，預設: 全部)'
    )
    
    parser.add_argument(
        '--environment',
        type=str,
        default='production',
        choices=['development', 'testing', 'staging', 'production'],
        help='執行環境 (預設: production)'
    )
    
    parser.add_argument(
        '--export-formats',
        nargs='+',
        default=['jsonl', 'csv'],
        choices=['jsonl', 'csv', 'arrow'],
        help='匯出格式 (預設: jsonl csv)'
    )
    
    parser.add_argument(
        '--disable-checkpointing',
        action='store_true',
        help='停用檢查點功能'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='日誌級別 (預設: INFO)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細輸出模式'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """驗證命令列參數"""
    # 檢查資料集路徑是否存在
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {args.dataset_path}")
    
    # 檢查批次大小
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive: {args.batch_size}")
    
    # 檢查最大記錄數
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError(f"Max records must be positive: {args.max_records}")


def print_system_info(args: argparse.Namespace) -> None:
    """印出系統資訊"""
    print("=" * 80)
    print("繁體中文程式碼問答資料集轉換系統 (Multi-Agent)")
    print("Traditional Chinese Code-QA Dataset Conversion System")
    print("=" * 80)
    print(f"資料集路徑: {args.dataset_path}")
    print(f"資料集類型: {args.dataset_type}")
    print(f"輸出目錄: {args.output_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"執行環境: {args.environment}")
    print(f"匯出格式: {', '.join(args.export_formats)}")
    if args.max_records:
        print(f"最大記錄數: {args.max_records}")
    print(f"檢查點: {'停用' if args.disable_checkpointing else '啟用'}")
    print("-" * 80)


def run_dataset_conversion(args: argparse.Namespace) -> None:
    """執行資料集轉換"""
    logger = logging.getLogger(__name__)
    
    try:
        # 創建資料集管理器
        dataset_manager = DatasetManager(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            enable_checkpointing=not args.disable_checkpointing
        )
        
        logger.info("開始資料集處理...")
        
        # 執行資料集處理
        batch_report = dataset_manager.process_dataset(
            dataset_path=args.dataset_path,
            dataset_type=args.dataset_type,
            max_records=args.max_records
        )
        
        logger.info("資料集處理完成，開始匯出結果...")
        
        # 匯出處理後的資料 (這裡需要從資料集管理器取得記錄)
        # 由於當前實作中沒有直接返回記錄列表，我們先記錄完成狀態
        print("\n" + "=" * 80)
        print("處理結果摘要")
        print("=" * 80)
        print(f"總記錄數: {batch_report.total_records}")
        print(f"成功處理: {batch_report.passed_records}")
        print(f"處理失敗: {batch_report.failed_records}")
        print(f"成功率: {batch_report.get_success_rate():.2%}")
        print(f"失敗率: {batch_report.get_failure_rate():.2%}")
        print(f"平均品質分數: {batch_report.average_quality_score:.2f}")
        print(f"處理時間: {batch_report.total_processing_time:.2f} 秒")
        print(f"平均處理時間: {batch_report.average_processing_time:.2f} 秒/記錄")
        
        # 匯出品質報告
        exporter = DatasetExporter()
        report_path = Path(args.output_dir) / "quality_report.json"
        exporter.converter.export_quality_report(batch_report, report_path)
        
        print(f"\n品質報告已匯出到: {report_path}")
        print("=" * 80)
        
        logger.info("資料集轉換完成!")
        
    except Exception as e:
        logger.error(f"資料集轉換失敗: {e}")
        sys.exit(1)


def main() -> None:
    """主程式入口"""
    try:
        # 載入環境變數
        load_dotenv()
        
        # 解析命令列參數
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        # 驗證參數
        validate_arguments(args)
        
        # 設定環境
        set_environment(args.environment)
        
        # 設定日誌
        setup_logging(
            log_level=args.log_level,
            verbose=args.verbose
        )
        
        # 印出系統資訊
        print_system_info(args)
        
        # 執行資料集轉換
        run_dataset_conversion(args)
        
    except KeyboardInterrupt:
        print("\n程式被使用者中斷")
        sys.exit(1)
    except Exception as e:
        print(f"錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

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


def setup_argument_parser() -> argparse.ArgumentParser:
    """設定命令列參數解析器"""
    parser = argparse.ArgumentParser(
        description='繁體中文程式碼問答資料集轉換系統 (Multi-Agent)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  # 處理 OpenCoder 教育指導資料集
  uv run python main.py --dataset-path data/opencoder_dataset_educational_instruct --dataset-type opencoder --output-dir output/educational_instruct
  
  # 處理 Evol Instruct 資料集
  uv run python main.py --dataset-path data/opencoder_dataset_evol_instruct --dataset-type opencoder --output-dir output/evol_instruct

  # 測試模式 (只處理前 10 筆記錄)
  uv run python main.py --dataset-path data/opencoder_dataset_educational_instruct --output-dir output/test_run --max-records 10 --environment development

  # 恢復處理（從中斷處繼續）
  uv run python main.py --dataset-path data/opencoder_dataset_package_instruct --output-dir output/package_instruct --resume
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
        '--resume',
        action='store_true',
        help='恢復處理（檢查已處理記錄，重跑失敗和缺失的）'
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
    print(f"執行環境: {args.environment}")
    print(f"匯出格式: {', '.join(args.export_formats)}")
    if args.max_records:
        print(f"最大記錄數: {args.max_records}")
    print(f"檢查點: {'停用' if args.disable_checkpointing else '啟用'}")
    if args.resume:
        print("模式: 恢復處理（重跑失敗和缺失記錄）")
    else:
        print("模式: 全新處理（逐筆處理並立即保存）")
    print("-" * 80)


def run_dataset_conversion(args: argparse.Namespace) -> None:
    """執行資料集轉換"""
    logger = logging.getLogger(__name__)
    
    try:
        # 創建資料集管理器
        dataset_manager = DatasetManager(
            output_dir=args.output_dir,
            enable_checkpointing=not args.disable_checkpointing
        )
        
        logger.info("開始資料集處理...")
        
        # 根據參數選擇處理模式
        if args.resume:
            logger.info("恢復處理模式...")
            batch_report = dataset_manager.resume(
                dataset_path=args.dataset_path,
                dataset_type=args.dataset_type,
                max_records=args.max_records
            )
        else:
            logger.info("全新處理模式...")
            batch_report = dataset_manager.run(
                dataset_path=args.dataset_path,
                dataset_type=args.dataset_type,
                max_records=args.max_records
            )
        
        logger.info("資料集處理完成！")
        
        # 顯示處理結果摘要
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
        
        # 顯示當前狀態
        print("\n" + "-" * 40)
        print("最終狀態報告")
        print("-" * 40)
        # 打印恢復狀態
        recovery_status = dataset_manager.recovery_manager.get_processed_status()
        print(f"已處理成功記錄: {len(recovery_status['successful'])}")
        print(f"處理失敗記錄: {len(recovery_status['failed'])}")
        print(f"總計處理記錄: {recovery_status['total']}")
        
        # 結果文件位置
        print(f"\n處理記錄已保存到: {Path(args.output_dir) / 'processed_records.jsonl'}")
        print(f"品質報告已保存到: {Path(args.output_dir) / 'quality_report.json'}")
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

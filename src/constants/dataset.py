"""
Dataset-related constants and configurations
"""

from typing import Dict, List

# Supported file formats for dataset loading
SUPPORTED_FORMATS = {
    "arrow": [".arrow"],
    "jsonl": [".jsonl", ".json"],
    "csv": [".csv"],
    "parquet": [".parquet"],
}

# Dataset types from OpenCoder
DATASET_TYPES = {
    "educational_instruct": "opencoder_dataset_educational_instruct",
    "evol_instruct": "opencoder_dataset_evol_instruct", 
    "mceval_instruct": "opencoder_dataset_mceval_instruct",
    "package_instruct": "opencoder_dataset_package_instruct",
}

# Processing batch configuration
DEFAULT_BATCH_SIZE = 100
MIN_BATCH_SIZE = 10
MAX_BATCH_SIZE = 1000

# Memory management thresholds
MEMORY_THRESHOLDS = {
    "warning_threshold_mb": 2048,  # 2GB warning
    "critical_threshold_mb": 4096,  # 4GB critical
    "batch_size_reduction_factor": 0.5,
}

# Progress tracking intervals
PROGRESS_SAVE_INTERVAL = 100  # Save progress every N records
PROGRESS_LOG_INTERVAL = 10    # Log progress every N records

# File encoding settings
DEFAULT_ENCODING = "utf-8"
FALLBACK_ENCODINGS = ["utf-8", "utf-16", "gbk", "big5"]

# Dataset schema validation
REQUIRED_FIELDS = {
    "instruction": str,
    "output": str,
}

OPTIONAL_FIELDS = {
    "id": str,
    "source": str,
    "metadata": dict,
}

# Export format configurations
EXPORT_FORMATS = {
    "jsonl": {
        "extension": ".jsonl",
        "encoding": "utf-8",
        "ensure_ascii": False,
    },
    "arrow": {
        "extension": ".arrow",
        "compression": "lz4",
    },
    "csv": {
        "extension": ".csv", 
        "encoding": "utf-8",
        "index": False,
    },
}

# Dataset processing stages
PROCESSING_STAGES = [
    "data_loading",
    "translation", 
    "qa_execution",
    "evaluation",
    "quality_check",
    "output_generation",
    "completed",
]

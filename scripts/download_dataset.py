#!/usr/bin/env python3
"""
Download and explore the QuixiAI/OpenCoder-LLM_opc-sft-stage2-DolphinLabeled dataset
"""

import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def download_dataset():
    """Download the QuixiAI/OpenCoder-LLM_opc-sft-stage2-DolphinLabeled dataset"""
    
    print("Downloading QuixiAI/OpenCoder-LLM_opc-sft-stage2-DolphinLabeled dataset...")
    
    # Available configs: educational_instruct, evol_instruct, mceval_instruct, package_instruct
    configs = ['educational_instruct', 'evol_instruct', 'mceval_instruct', 'package_instruct']
    datasets = {}
    
    try:
        for config in configs:
            print(f"\nDownloading config: {config}")
            # Load the dataset from Hugging Face
            dataset = load_dataset(
                "QuixiAI/OpenCoder-LLM_opc-sft-stage2-DolphinLabeled",
                config
            )
            datasets[config] = dataset
            
            print(f"Dataset {config} loaded successfully!")
            print(f"Dataset info: {dataset}")
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save each dataset config locally
        for config, dataset in datasets.items():
            dataset.save_to_disk(f"data/opencoder_dataset_{config}")
            print(f"Dataset {config} saved to data/opencoder_dataset_{config}/")
        
        # Explore the first dataset structure as example
        print(f"\nExploring first config ({configs[0]}) as example:")
        explore_dataset(datasets[configs[0]])
        
        return datasets
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def explore_dataset(dataset):
    """Explore the structure and content of the dataset"""
    
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    
    # Print dataset splits
    print(f"Available splits: {list(dataset.keys())}")
    
    for split_name, split_data in dataset.items():
        print(f"\n--- {split_name.upper()} SPLIT ---")
        print(f"Number of examples: {len(split_data)}")
        print(f"Features: {split_data.features}")
        
        # Show first few examples
        if len(split_data) > 0:
            print(f"\nFirst example:")
            first_example = split_data[0]
            for key, value in first_example.items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"{key}: {value[:200]}...")
                else:
                    print(f"{key}: {value}")
    
    # Convert to pandas for easier analysis
    if 'train' in dataset:
        train_df = dataset['train'].to_pandas()
        print(f"\n--- DATASET STATISTICS ---")
        print(f"Training set shape: {train_df.shape}")
        print(f"Columns: {list(train_df.columns)}")
        
        # Save sample to CSV for inspection
        sample_df = train_df.head(100)
        sample_df.to_csv("data/dataset_sample.csv", index=False)
        print("Sample of 100 examples saved to data/dataset_sample.csv")

def main():
    """Main function to download and explore the dataset"""
    
    print("TW Code QA Dataset Creation Project")
    print("Base dataset: QuixiAI/OpenCoder-LLM_opc-sft-stage2-DolphinLabeled")
    print("="*60)
    
    dataset = download_dataset()
    
    if dataset:
        print("\n✅ Dataset download and exploration completed successfully!")
        print(f"Downloaded {len(dataset)} configurations:")
        for config in dataset.keys():
            print(f"  - {config}")
        print("\nNext steps:")
        print("1. Review the dataset structure and content")
        print("2. Set up LangGraph multi-agent system for Traditional Chinese evaluation")
        print("3. Create quality assessment workflow")
    else:
        print("\n❌ Failed to download dataset")

if __name__ == "__main__":
    main()

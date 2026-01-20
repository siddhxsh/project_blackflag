"""
Utility functions for consistent file handling across scripts.
"""
import os
from pathlib import Path


def get_base_dir():
    """Get the MLmodel directory path."""
    return Path(__file__).resolve().parents[0].parent


def find_cleaned_csv(data_folder=None):
    """
    Auto-detect cleaned CSV in data folder.
    
    Args:
        data_folder: Optional custom data folder path
        
    Returns:
        Path to first *_cleaned.csv found
        
    Raises:
        FileNotFoundError: If no cleaned CSV found
    """
    if data_folder is None:
        data_folder = get_base_dir() / "data"
    else:
        data_folder = Path(data_folder)
    
    cleaned_files = sorted(data_folder.glob("*_cleaned.csv"))
    
    if not cleaned_files:
        raise FileNotFoundError(
            f"No cleaned CSV files (*_cleaned.csv) found in {data_folder}. "
            f"Run cleaning.py first."
        )
    
    return cleaned_files[0]


def get_model_dir():
    """Get the models directory path."""
    return get_base_dir() / "models"

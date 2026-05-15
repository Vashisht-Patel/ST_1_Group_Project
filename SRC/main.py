""" 
******************************* 
Software Technology 1 
Assessment 3 – Part A 
Group: X 
Authors: 
u3330114 
u3258243 
u3298551 
Date: 
15/05/2026 
******************************* 
"""

'''
*******************************
Author: u3298551
Group: 3
Assessment: 3
Date: 12/05/2026
Programming: Compacted main file, linked it to the GUI and eda service
*******************************
'''


import config
from GUI import MacroApp
from pandas import Series
from Services.dataset_indexer import DatasetIndexer
from Services.eda_service import EDAService
from typing import Tuple
import pandas as pd
import os
from pathlib import Path

## RUN GUI
def main():
    # SET GUI       files_to_display        update_progress
    gui = MacroApp(config.EDA_OUTPUT_DIR, start_callback=load_dataset)
    gui.mainloop()


## LOAD DATASET
def load_dataset(progress_callback=None) -> Tuple[pd.DataFrame, Path]:
    # Declare variables
    output_dir: Path
    class_counts: Series

    # Initialize data indexing object
    indexer = DatasetIndexer(config.RAW_DATA_DIR)

    output_dir = Path(config.EDA_OUTPUT_DIR)
    dataset_dataframe = indexer.build_dataframe(progress_callback=progress_callback)
    required_columns = [
        "image_path",
        "label",
        "width",
        "height",
    ]
    for col in required_columns:
        if col not in dataset_dataframe.columns:
            raise ValueError(f"Missing required column: {col}")

    class_counts = dataset_dataframe["label"].value_counts()

    print("Dataset Loaded & Indexed Successfully")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    return dataset_dataframe, output_dir

if __name__ == "__main__":
    main()


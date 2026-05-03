import config
from GUI import MacroApp

from Services.dataset_indexer import DatasetIndexer
from Services.eda_service import EDAService

import os
from pathlib import Path


# Load dataset
def load_dataset():
    # Declare variables
    output_dir: Path
    class_counts: Series

    # Initialize data indexing object
    indexer = DatasetIndexer(config.RAW_DATA_DIR)

    output_dir = Path(config.EDA_OUTPUT_DIR)
    indexer = DatasetIndexer(config.RAW_DATA_DIR)
    dataset_dataframe = indexer.build_dataframe()
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

    # Initialize EDAService object
    eda_service = EDAService(dataframe=dataset_dataframe, output_dir=output_dir)
    eda_service.plot_class_distribution()
    eda_service.plot_image_sizes()

    print("Visualizations generated")
    print(eda_service.build_summary())

def main():
    # SET GUI
    gui = MacroApp(None, f"{config.MODEL_OUTPUT_DIR}/macroinvertebrates_classifier.h5")

    # START GUI

    # LET USER SELECT DATASET DIR, WHAT VISUALISATIONS TO USE

    # AWAIT USER PRESSING "START BUTTON"

    # CONFIRM ALL NECESSARY OPTIONS ARE SELECTED AND VALID

    # RUN DATA ANALYSIS
    load_dataset()

    # INTEGRATE CHARTS INTO GUI

    # ALLOW USER TO REPEAT PROCESS

if __name__ == "__main__":
    main()


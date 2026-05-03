from Services.dataset_indexer import DatasetIndexer
from Services.eda_service import EDAService
import os
from pathlib import Path

DATASET_PATH = "Data/"
VISUALIZED_DATA_PATH = "./eda_outputs"


# Load dataset
def load_dataset():
    output_dir: Path
    class_counts: Series

    # Initialize data indexing object
    indexer = DatasetIndexer(DATASET_PATH)

    output_dir = Path(VISUALIZED_DATA_PATH)
    class_counts = None
    eda_service = None
    indexer = DatasetIndexer(DATASET_PATH)
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
    # Run everything
    load_dataset()

if __name__ == "__main__":
    main()


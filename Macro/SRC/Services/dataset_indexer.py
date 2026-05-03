from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

# Images with different extensions will be ignored
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

class DatasetIndexer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def build_dataframe(self):
        records = []

        # Collect all files first so tqdm can show total progress
        all_files = [
            file_path
            for file_path in self.data_dir.rglob("*")
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        for file_path in tqdm(all_files, desc="Indexing images"):
            image = cv2.imread(str(file_path))

            if image is None:
                continue

            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1

            label = file_path.parent.name

            records.append({
                "image_path": str(file_path),
                "label": label,
                "width": width,
                "height": height,
                "channels": channels,
            })

        return pd.DataFrame(records)

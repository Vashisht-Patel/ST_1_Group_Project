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


import config
from Models.records import ImageRecord
import numpy as np
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

class DatasetIndexer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def build_dataframe(self, progress_callback=None):
        records = []

        # Collect all files first so tqdm can show total progress
        all_files = [
            file_path
            for file_path in self.data_dir.rglob("*")
            if file_path.suffix.lower() in config.SUPPORTED_EXTENSIONS
        ]

        total_files = len(all_files)

        for index, file_path in enumerate(all_files):

            image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

            if image is None:
                continue

            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1

            label = file_path.parent.name

            records.append(
                ImageRecord(
                    image_path=file_path,
                    label=label,
                    width=width,
                    height=height,
                    channels=channels,
                    file_extension=file_path.suffix.lower(),
                    aspect_ratio=width / height
                )
            )

            if progress_callback:
                progress_callback(index + 1, total_files)

        return pd.DataFrame(records)

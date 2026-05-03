from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# NOT CURRENTLY IN USE

# Save a grid of sample images for quick visual inspection.
def save_sample_grid(dataframe: pd.DataFrame, output_path: Path, sample_count: int = 9) -> None:
    sample_df = dataframe.sample(min(sample_count, len(dataframe)),
    random_state=42)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for ax, (_, row) in zip(axes.flat, sample_df.iterrows()):
        image = cv2.imread(row["file_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title(row["label"])
        ax.axis("off")
    for ax in axes.flat[len(sample_df):]:
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

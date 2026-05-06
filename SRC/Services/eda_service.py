from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import seaborn as sns
from pandas import Series
from typing import List

class EDAService:
    # Generate and save EDA outputs for the indexed image dataset.
    def __init__(self, dataframe: pd.DataFrame, output_dir: Path) -> None:
        self.dataframe = dataframe
        self.output_dir = output_dir
        self.class_counts = dataframe["label"].value_counts()

    # Class distribution       Use different coloured bars to show how different the number of images is between classes (based on median class num)
    #   Balanced = within 20% of median?        Not balanced = 50% smaller or 200% larger than median,      Anything inbetween is mildly unbalanced
    def plot_class_distribution(self, save: bool = True):
        # Declare variables
        counts: Series
        classes: List[str] = []
        colours: List[str] = []
        values: np.ndarray
        total: int
        median: float

        counts = self.class_counts.sort_values(ascending=True)
        classes = counts.index.tolist()
        values = counts.values
        total = values.sum()
        median = np.median(values)

        BALANCED_COLOUR = "green"
        SLIGHTLY_UNBALANCED_COLOUR = "orange"
        VERY_UNBALANCED_COLOUR = "red"

        # Determine the correct colour for every class
        for value in values:

            # Balanced class
            is_balanced = ((median * 0.8) <= value <= (median * 1.2))

            # Very unbalanced class
            is_very_unbalanced = ((value < (median * 0.5)) or (value > (median * 2.0)))

            # Choose the colour based on the category
            if is_balanced:
                selected_colour = BALANCED_COLOUR

            elif is_very_unbalanced:
                selected_colour = VERY_UNBALANCED_COLOUR

            else:
                selected_colour = SLIGHTLY_UNBALANCED_COLOUR

            # Store the chosen colour for this bar
            colours.append(selected_colour)

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.5), 6))

        fig.canvas.manager.set_window_title("Class Distribution Visualization")

        # Create a horizontal bar chart using the calculated colours
        bars = ax.barh(classes, values, color=colours)

        # Add labels and title
        ax.set_xlabel("Image Count")
        ax.set_title("Class Distribution")

        legend_elements = [
            # Needs to use american spelling
            patches.Patch(color=BALANCED_COLOUR, label="Balanced (within 20% of median)"),
            patches.Patch(color=SLIGHTLY_UNBALANCED_COLOUR, label="Mildly unbalanced"),
            patches.Patch(color=VERY_UNBALANCED_COLOUR, label="Not balanced (< 50% or > 200% of median)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
        ax.axvline(median, color="gray", linestyle="--", linewidth=1, label=f"Median: {median:.0f}")   # Median line

        plt.tight_layout()

        # If user wants to save the visualisation
        if save:
            out_path = self.output_dir / "class_distribution.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved class distribution to: {out_path}")

        plt.show()
        plt.close(fig)

    # Image size visualisation (Coloured rectangle outlines showcase any obvious size differences)
    def plot_image_sizes(self, save: bool = True):
        # IMPLEMENT!!

        if save:
            out_path = self.output_dir / "image_sizes.png"
            # fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved image size overlay to: {out_path}")

        # plt.show()
        # plt.close(fig)

    # Return key dataset summary statistics.
    def build_summary(self) -> dict[str, float]:
        return {
            "total_images": int(len(self.dataframe)),
            "total_classes": int(self.dataframe["label"].nunique()),
            "mean_width": float(self.dataframe["width"].mean()),
            "mean_height": float(self.dataframe["height"].mean()),
        }

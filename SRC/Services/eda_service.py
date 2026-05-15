import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from pandas import Series
from typing import List
import cv2

import config

'''
*******************************
Author: u3298551
Group: 3
Assessment: 3
Date: 04/05/2026
Programming: Created EDA class and functions
*******************************
'''

class EDAService:
    # Generate and save EDA outputs for the indexed image dataset.
    def __init__(self, dataframe: pd.DataFrame, output_dir: Path) -> None:
        self.dataframe = dataframe
        config.OUTPUTS_DIR = output_dir
        self.class_counts = dataframe["label"].value_counts()
        self.readable_dataframe = dataframe     # DOUBLE CHECK

    # Set output generation functions
    def generate_all_outputs(self) -> list[Path]:
        if self.dataframe.empty:
            raise ValueError("The dataset index is empty.")

        output_paths = [
            self.generate_dataset_summary(),
            self.plot_class_distribution(),
            self.generate_image_size_distribution_chart(),
            self.generate_width_height_scatter_plot(),
            self.generate_sample_image_grid(),
            self.generate_width_by_class_boxplot(),
            self.generate_height_by_class_boxplot(),
            self.generate_pixel_intensity_histogram(),
            self.generate_class_imbalance_report()
        ]

        return output_paths

    # Class distribution       Use different coloured bars to show how different the number of images is between classes (based on median class num)
    #   Balanced = within 20% of median?        Not balanced = 50% smaller or 200% larger than median,      Anything inbetween is mildly unbalanced
    def plot_class_distribution(self, save: bool = True) -> None:
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
            out_path = config.OUTPUTS_DIR / "class_distribution.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved class distribution to: {out_path}")


    """
    Some functions in this file are adapted from example code provided by the course tutor.
    The implementation has been modified while retaining similar functionality.
    """

    # txt file to complement class distribution
    def generate_class_imbalance_report(self) -> Path:
        class_counts: Series
        largest_class: str
        smallest_class: str
        largest_count: int
        smallest_count: int
        ratio: float
        explanation: str
        report: List[str]
        out_path: Path

        class_counts = self.dataframe["label"].value_counts().sort_values(
            ascending=False
        )
        largest_class = class_counts.idxmax()
        smallest_class = class_counts.idxmin()
        largest_count = int(class_counts.max())
        smallest_count = int(class_counts.min())
        ratio = largest_count / smallest_count if smallest_count > 0 else float("inf")

        explanation = (
            "The class balance should be considered before any future Stage 2 "
            "classification work. A high imbalance ratio can bias a model towards "
            "the most common class and make minority macroinvertebrate classes "
            "harder to recognise."
        )

        report = [
            "# Class Imbalance Report:",
            "",
            f"- Largest class: **{largest_class}** ({largest_count} images)",
            f"- Smallest class: **{smallest_class}** ({smallest_count} images)",
            f"- Imbalance ratio: **{ratio:.2f}:1**",
            "",
            "## Interpretation:",
            "",
            explanation
        ]

        out_path = config.OUTPUTS_DIR / "class_distribution.txt"
        out_path.write_text("\n".join(report), encoding="utf-8")
        return out_path


    # GENERATE CSV SUMMARY
    def generate_dataset_summary(self) -> Path:
        class_counts: Series
        readable: pd.DataFrame
        supported_types: str
        summary_rows: list
        summary: pd.DataFrame
        output_path: Path

        class_counts = self.dataframe["label"].value_counts().sort_index()
        readable = self.readable_dataframe
        supported_types = ", ".join(sorted(self.dataframe["file_extension"].unique()))

        summary_rows = [
            ("total_images", len(self.dataframe)),
            ("total_classes", self.dataframe["label"].nunique()),
            ("images_per_class", self._format_class_counts(class_counts)),
            ("mean_width", self._safe_round(readable["width"].mean())),
            ("mean_height", self._safe_round(readable["height"].mean())),
            ("min_width", self._safe_int(readable["width"].min())),
            ("max_width", self._safe_int(readable["width"].max())),
            ("min_height", self._safe_int(readable["height"].min())),
            ("max_height", self._safe_int(readable["height"].max())),
            ("supported_file_types_found", supported_types),
        ]

        summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
        output_path = config.EDA_OUTPUT_DIR / "dataset_summary.csv"
        summary.to_csv(output_path, index=False)
        return output_path

    ## IMAGE SIZE COMPARISON ##
    def generate_image_size_distribution_chart(self) -> Path:
        readable: pd.DataFrame
        fig: plt.Figure
        axes: np.ndarray
        output_path: Path

        readable = self._require_readable_images()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(readable["width"], bins=20, ax=axes[0], color="#4C78A8")
        axes[0].set_title("Image Width Distribution")
        axes[0].set_xlabel("Width in pixels")

        sns.histplot(readable["height"], bins=20, ax=axes[1], color="#F58518")
        axes[1].set_title("Image Height Distribution")
        axes[1].set_xlabel("Height in pixels")

        fig.tight_layout()
        output_path = config.EDA_OUTPUT_DIR / "image_size_distribution.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def generate_width_height_scatter_plot(self) -> Path:
        readable: pd.DataFrame
        fig: plt.Figure
        ax: plt.Axes
        classes: List[str]
        palette: List
        class_counts: Series
        axis_max: float
        output_path: Path

        readable = self._require_readable_images()
        classes = sorted(readable["label"].unique().tolist())
        palette = sns.color_palette("tab10", n_colors=len(classes))
        class_counts = readable["label"].value_counts()

        fig, ax = plt.subplots(figsize=(9, 7))

        # Draw each class with a distinct colour and marker shape, cycling through available markers
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", ">"]
        for index, label in enumerate(classes):
            subset = readable[readable["label"] == label]
            count = class_counts[label]
            ax.scatter(
                subset["width"],
                subset["height"],
                label=f"{label} (n={count})",
                color=palette[index],
                marker=markers[index % len(markers)],
                alpha=0.6,
                s=40,
                edgecolors="none",
            )

        # Unity line
        axis_max = float(max(readable["width"].max(), readable["height"].max()) * 1.05)
        ax.plot(
            [0, axis_max],
            [0, axis_max],
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Square (width = height)",
        )

        ax.set_title("Image Width Versus Height")
        ax.set_xlabel("Width in pixels")
        ax.set_ylabel("Height in pixels")
        ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()

        output_path = config.EDA_OUTPUT_DIR / "width_height_scatter.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def generate_sample_image_grid(self) -> Path:
        readable: pd.DataFrame
        samples: pd.DataFrame
        columns: int
        rows: int
        fig: plt.Figure
        axes_array: np.ndarray
        output_path: Path

        readable = self._require_readable_images()
        samples = self._select_representative_samples(readable)

        columns = min(4, len(samples))
        rows = int(np.ceil(len(samples) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.4 * rows))
        axes_array = np.array(axes).reshape(-1)

        for axis in axes_array:
            axis.axis("off")

        for axis, (_, row) in zip(axes_array, samples.iterrows()):
            image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axis.imshow(rgb_image)
            axis.set_title(str(row["label"]), fontsize=10)
            axis.axis("off")

        fig.suptitle("Representative Sample Images by Class")
        fig.tight_layout()
        output_path = config.EDA_OUTPUT_DIR / "sample_image_grid.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def generate_width_by_class_boxplot(self) -> Path:
        readable: pd.DataFrame
        fig: plt.Figure
        ax: plt.Axes
        classes: List[str]
        palette: List
        class_order: List[str]
        dataset_median: float
        class_counts: Series
        output_path: Path

        readable = self._require_readable_images()
        classes = sorted(readable["label"].unique().tolist())
        palette = sns.color_palette("tab10", n_colors=len(classes))
        class_counts = readable["label"].value_counts()

        # Sort x axis by median width so the chart reads as a ranking
        class_order = (
            readable.groupby("label")["width"]
            .median()
            .sort_values()
            .index.tolist()
        )

        dataset_median = float(readable["width"].median())

        fig, ax = plt.subplots(figsize=(max(11, len(classes) * 1.1), 6))

        sns.boxplot(
            data=readable,
            x="label",
            y="width",
            order=class_order,
            palette={label: palette[index] for index, label in enumerate(classes)},
            ax=ax,
            width=0.5,
            flierprops={"marker": "x", "markersize": 4, "alpha": 0.4},
        )

        # Strip plot overlay to reveal individual point density per class
        sns.stripplot(
            data=readable,
            x="label",
            y="width",
            order=class_order,
            palette={label: palette[index] for index, label in enumerate(classes)},
            ax=ax,
            size=3,
            alpha=0.3,
            jitter=True,
            dodge=False,
        )

        # Dataset-wide median reference line
        ax.axhline(
            dataset_median,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Dataset median: {dataset_median:.0f}px",
        )

        # Annotate each class box with its sample count
        for index, label in enumerate(class_order):
            count = class_counts[label]
            ax.text(
                index,
                ax.get_ylim()[0],
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="dimgray",
            )

        ax.set_title("Image Width by Class")
        ax.set_xlabel("Class label")
        ax.set_ylabel("Width in pixels")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()

        output_path = config.EDA_OUTPUT_DIR / "width_by_class_boxplot.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def generate_height_by_class_boxplot(self) -> Path:
        readable: pd.DataFrame
        fig: plt.Figure
        ax: plt.Axes
        classes: List[str]
        palette: List
        class_order: List[str]
        dataset_median: float
        class_counts: Series
        output_path: Path

        readable = self._require_readable_images()
        classes = sorted(readable["label"].unique().tolist())
        palette = sns.color_palette("tab10", n_colors=len(classes))
        class_counts = readable["label"].value_counts()

        # Sort x axis by median height so the chart reads as a ranking
        class_order = (
            readable.groupby("label")["height"]
            .median()
            .sort_values()
            .index.tolist()
        )

        dataset_median = float(readable["height"].median())

        fig, ax = plt.subplots(figsize=(max(11, len(classes) * 1.1), 6))

        sns.boxplot(
            data=readable,
            x="label",
            y="height",
            order=class_order,
            palette={label: palette[index] for index, label in enumerate(classes)},
            ax=ax,
            width=0.5,
            flierprops={"marker": "x", "markersize": 4, "alpha": 0.4},
        )

        # Strip plot overlay to reveal individual point density per class
        sns.stripplot(
            data=readable,
            x="label",
            y="height",
            order=class_order,
            palette={label: palette[index] for index, label in enumerate(classes)},
            ax=ax,
            size=3,
            alpha=0.3,
            jitter=True,
            dodge=False,
        )

        # Dataset-wide median reference line
        ax.axhline(
            dataset_median,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Dataset median: {dataset_median:.0f}px",
        )

        # Annotate each class box with its sample count
        for index, label in enumerate(class_order):
            count = class_counts[label]
            ax.text(
                index,
                ax.get_ylim()[0],
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="dimgray",
            )

        ax.set_title("Image Height by Class")
        ax.set_xlabel("Class label")
        ax.set_ylabel("Height in pixels")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()

        output_path = config.EDA_OUTPUT_DIR / "height_by_class_boxplot.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path


    # IMAGE QUALITY
    def generate_pixel_intensity_histogram(self) -> Path:
        readable: pd.DataFrame
        sample: pd.DataFrame
        intensity_values: List[int]
        fig: plt.Figure
        ax: plt.Axes
        output_path: Path

        readable = self._require_readable_images()
        sample = readable.head(config.PIXEL_ANALYSIS_SAMPLE_SIZE)
        intensity_values = []

        for _, row in sample.iterrows():
            grayscale = cv2.imread(str(row["image_path"]), cv2.IMREAD_GRAYSCALE)
            if grayscale is not None:
                intensity_values.extend(grayscale.flatten().tolist())

        if not intensity_values:
            raise ValueError("No readable pixels were available for intensity analysis.")

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.histplot(intensity_values, bins=50, color="#B279A2", ax=ax)
        ax.set_title("Sampled Grayscale Pixel Intensity Distribution")
        ax.set_xlabel("Pixel intensity, 0 dark to 255 bright")
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        output_path = config.EDA_OUTPUT_DIR / "pixel_intensity_histogram.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    ## HELPER FUNCTIONS ##
    # Double check image supplied is valid
    def _require_readable_images(self) -> pd.DataFrame:
        if self.readable_dataframe.empty:
            raise ValueError("No readable images were found for EDA charts.")

        return self.readable_dataframe

    # Generate "samples" (set number of images to display) for each class
    def _select_representative_samples(self, readable: pd.DataFrame) -> pd.DataFrame:
        per_class: pd.DataFrame
        remaining_slots: int
        remaining: pd.DataFrame

        per_class = readable.groupby("label", group_keys=False).head(1)
        if len(per_class) >= config.SAMPLE_GRID_MAX_IMAGES:
            return per_class.head(config.SAMPLE_GRID_MAX_IMAGES)

        remaining_slots = config.SAMPLE_GRID_MAX_IMAGES - len(per_class)
        remaining = readable.drop(per_class.index).head(remaining_slots)
        return pd.concat([per_class, remaining])

    def _format_class_counts(self, class_counts: pd.Series) -> str:
        return "; ".join(
            f"{label}: {count}" for label, count in class_counts.items()
        )

    def _safe_round(self, value: float) -> float:
        if pd.isna(value):
            return 0.0

        return round(float(value), 2)

    # Convert to int
    def _safe_int(self, value: float) -> int:
        if pd.isna(value):
            return 0

        return int(value)


    # Return key dataset summary statistics.
    def build_summary(self) -> dict[str, float]:
        return {
            "total_images": int(len(self.dataframe)),
            "total_classes": int(self.dataframe["label"].nunique()),
            "mean_width": float(self.dataframe["width"].mean()),
            "mean_height": float(self.dataframe["height"].mean()),
        }

    def generate_dataset_warnings(self) -> list[str]:
        warnings: List[str]
        counts: Series
        low_res_count: int

        warnings = []
        counts = self.class_counts

        if counts.max() > counts.min() * 5:
            warnings.append(
                "Dataset imbalance detected: some classes contain significantly more images than others."
            )
        low_res_count = len(
            self.dataframe[
            (self.dataframe["width"] < 100)
            | (self.dataframe["height"] < 100)
            ]
        )

        if low_res_count > 0:
            warnings.append(
            f"{low_res_count} low-resolution images detected."
        )

        return warnings

from dataclasses import dataclass
from pathlib import Path

@dataclass
class ImageRecord:
    # Store the core metadata for one indexed macroinvertebrate image.
    image_path: Path
    label: str
    width: int
    height: int
    channels: int

    file_extension: str
    aspect_ratio: float

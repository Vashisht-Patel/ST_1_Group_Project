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
Programming: Updated record to account for more variables
*******************************
'''


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

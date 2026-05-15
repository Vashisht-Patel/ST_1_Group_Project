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


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
RAW_DATA_DIR = DATA_DIR / "Raw"
OUTPUTS_DIR = BASE_DIR / "Outputs"
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
MODEL_OUTPUT_DIR = OUTPUTS_DIR / "Models"
IMAGE_SIZE = (128, 128)
# Images with different extensions will be ignored
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

PIXEL_ANALYSIS_SAMPLE_SIZE = 100
SAMPLE_GRID_MAX_IMAGES     = 16

from pathlib import Path

'''
*******************************
Author: u3298551
Group: 3
Assessment: 3
Date: 04/05/2026
Programming: Created and updated config to correspond with project files
*******************************
'''

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

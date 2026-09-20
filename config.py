"""
SynthQC Configuration Module
Centralized configuration for dataset paths, hardware settings, and model hyperparameters.
"""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MVTEC_DIR = DATA_DIR / "mvtec"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = BASE_DIR / "saved_models"
CACHE_DIR = BASE_DIR / ".cache"

for d in [DATA_DIR, MVTEC_DIR, SYNTHETIC_DIR, MODELS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Product Category (Configurable: default is 'bottle' from MVTec AD)
PRODUCT_CATEGORY = "bottle"

# Hugging Face Mirror Configuration
HF_DATASET_REPO = "foersben/mvtec-ad"

# Alternative category download link from MVTec official site
MVTEC_OFFICIAL_URL_TEMPLATE = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937451-1629951672/{category}.tar.xz"

# Hardware Configuration (auto-detect GPU, fallback to CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image resolution settings
IMAGE_SIZE_CLASSIFIER = (224, 224)  # For ResNet18 fine-tuning & inspection
IMAGE_SIZE_GAN = (128, 128)         # For lightweight conditional GAN

# Subset limits for fast CPU execution
MAX_GOOD_TRAIN_SAMPLES = 60         # 40-80 images recommended for fast prototyping
MAX_GOOD_TEST_SAMPLES = 20
SYNTHETIC_DEFAULT_COUNT = 30

# Reproducibility
RANDOM_SEED = 42

# UI Styling Constants
THEME_PRIMARY = "#00F0FF"       # Cyber Cyan
THEME_SECONDARY = "#7928CA"     # Neon Purple
THEME_SUCCESS = "#00FF88"       # Emerald Pass
THEME_WARNING = "#FFB800"       # Amber Review
THEME_DANGER = "#FF3366"        # Ruby Defect
THEME_BG = "#0B0F19"            # Deep Space Navy
THEME_CARD = "#161E31"          # Frosted Surface

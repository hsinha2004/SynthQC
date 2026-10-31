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

# UI Styling Constants (Calibrated Industrial Metrology Theme)
THEME_PRIMARY = "#4A72B0"       # Technical Cobalt Steel
THEME_SECONDARY = "#374151"     # Industrial Graphite
THEME_SUCCESS = "#2E9E66"       # Calibrated Optical Pass
THEME_WARNING = "#D97706"       # Inspection Review Amber
THEME_DANGER = "#D33F49"        # Reject / Anomaly Red
THEME_BG = "#0D1117"            # Deep Graphite Slate
THEME_CARD = "#161B22"          # Matte Inspection Panel
THEME_BORDER = "rgba(255, 255, 255, 0.08)"
THEME_TEXT_PRIMARY = "#F0F6FC"
THEME_TEXT_SECONDARY = "#8B949E"
THEME_TEXT_MUTED = "#6E7681"

"""
SynthQC Data Manager Module (Rigorous Split & Zero Leakage Version)
Features:
- Deterministic, stratified fixed held-out test set (quarantined from all training/generation).
- Training defect pool strictly defined on remaining items.
- Support for 10%, 25%, 50%, 100% fractions of the remaining training pool.
- Multi-split / bootstrap seed generation for mean +/- std evaluation.
- Allotted-only patch extraction to prevent source leakage into synthetic generators.
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    MVTEC_DIR, SYNTHETIC_DIR, PRODUCT_CATEGORY, HF_DATASET_REPO,
    IMAGE_SIZE_CLASSIFIER, RANDOM_SEED
)


class SynthQCDataset(Dataset):
    """PyTorch Dataset for binary defect classification (0: Good, 1: Defective)."""
    def __init__(self, items: List[Tuple[str, int, str]], transform=None):
        """
        items: List of (image_path, label [0 or 1], defect_type_name)
        """
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label, defect_type = self.items[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", IMAGE_SIZE_CLASSIFIER, (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label, defect_type, str(img_path)


class DataManager:
    """
    Manages dataset verification, fixed held-out test sets, and strictly partitioned
    training defect pools with zero leakage.
    """
    def __init__(self, category: str = PRODUCT_CATEGORY):
        self.category = category
        self.category_dir = MVTEC_DIR / self.category
        self.synthetic_dir = SYNTHETIC_DIR / self.category
        self.category_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)

        self.train_good_dir = self.category_dir / "train" / "good"
        self.test_dir = self.category_dir / "test"
        self.gt_dir = self.category_dir / "ground_truth"
        self.registry_path = self.category_dir / "fixed_split_registry.json"

        # Initialize or load fixed stratified split
        self._init_fixed_splits()

    def _init_fixed_splits(self):
        """
        Carves out a permanent, stratified held-out test set once and for all.
        Stratified across:
        - 8 broken_large
        - 8 broken_small
        - 8 contamination
        - All available test good images (20)
        Remaining defects form the training defect pool (39 items).
        """
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                self.split_registry = json.load(f)
            return

        # Build fresh fixed split registry if dataset files exist
        if not (self.test_dir.exists() and self.gt_dir.exists()):
            self.split_registry = {"held_out_test": [], "training_pool": []}
            return

        rng = random.Random(RANDOM_SEED)

        held_out_test = []
        training_pool = []

        # 1. Good test images
        good_test_dir = self.test_dir / "good"
        if good_test_dir.exists():
            for f in sorted(list(good_test_dir.glob("*.png")) + list(good_test_dir.glob("*.jpg"))):
                held_out_test.append({
                    "img_path": str(f),
                    "mask_path": None,
                    "defect_type": "good",
                    "stem": f.stem,
                    "label": 0
                })

        # 2. Stratified defect subtypes (8 each held out)
        subtypes = ["broken_large", "broken_small", "contamination"]
        for st in subtypes:
            st_dir = self.test_dir / st
            gt_st_dir = self.gt_dir / st
            if not st_dir.exists():
                continue

            st_files = sorted(list(st_dir.glob("*.png")) + list(st_dir.glob("*.jpg")))
            st_items = []
            for f in st_files:
                mask_f = gt_st_dir / f"{f.stem}_mask.png"
                st_items.append({
                    "img_path": str(f),
                    "mask_path": str(mask_f) if mask_f.exists() else None,
                    "defect_type": st,
                    "stem": f.stem,
                    "label": 1
                })

            # Deterministic shuffle with seed
            rng.shuffle(st_items)
            held_out_count = min(8, len(st_items))
            held_out_items = st_items[:held_out_count]
            pool_items = st_items[held_out_count:]

            held_out_test.extend(held_out_items)
            training_pool.extend(pool_items)

        self.split_registry = {
            "held_out_test": held_out_test,
            "training_pool": training_pool
        }

        with open(self.registry_path, "w") as f:
            json.dump(self.split_registry, f, indent=2)

    def get_subtype_counts(self) -> Dict[str, Dict[str, int]]:
        """Returns exact counts of images per subtype in test vs training pool."""
        held_out = self.split_registry.get("held_out_test", [])
        pool = self.split_registry.get("training_pool", [])

        stats = {
            "held_out_test": {},
            "training_pool": {},
            "train_good": len(list(self.train_good_dir.glob("*.png"))) if self.train_good_dir.exists() else 0
        }

        for item in held_out:
            st = item["defect_type"]
            stats["held_out_test"][st] = stats["held_out_test"].get(st, 0) + 1

        for item in pool:
            st = item["defect_type"]
            stats["training_pool"][st] = stats["training_pool"].get(st, 0) + 1

        return stats

    def get_fixed_test_items(self) -> List[Tuple[str, int, str]]:
        """Returns the quarantined held-out test items: (img_path, label, defect_type)."""
        held_out = self.split_registry.get("held_out_test", [])
        return [(item["img_path"], item["label"], item["defect_type"]) for item in held_out]

    def sample_training_defects(self, fraction: float, split_seed: int) -> List[Dict]:
        """
        Samples a fraction (0.10, 0.25, 0.50, 1.0) of the REMAINING training defect pool
        using a specific seed.
        Returns the allotted defect items dicts: [{img_path, mask_path, defect_type, stem, label}].
        """
        pool = list(self.split_registry.get("training_pool", []))
        total_pool_size = len(pool)
        if total_pool_size == 0:
            return []

        # Stratified sampling across subtypes
        rng = random.Random(split_seed)
        subtypes = ["broken_large", "broken_small", "contamination"]
        sampled_items = []

        for st in subtypes:
            st_items = [it for it in pool if it["defect_type"] == st]
            rng.shuffle(st_items)
            count = max(1, int(round(len(st_items) * fraction))) if fraction < 1.0 else len(st_items)
            sampled_items.extend(st_items[:count])

        return sampled_items

    def get_train_good_images(self) -> List[str]:
        """Returns list of paths for good training images."""
        if not self.train_good_dir.exists():
            return []
        files = sorted(list(self.train_good_dir.glob("*.png")) + list(self.train_good_dir.glob("*.jpg")))
        return [str(f) for f in files]

    def build_run_dataset(
        self,
        allotted_real_defects: List[Dict],
        synthetic_defects: Optional[List[str]] = None,
        synthetic_label_name: str = "synthetic"
    ) -> List[Tuple[str, int, str]]:
        """
        Builds a training dataset for a single run using:
        1. All good training images (Label 0)
        2. Strictly the allotted real defects (Label 1)
        3. Optional synthetic defects generated solely from the allotted set (Label 1)
        """
        items = []

        # 1. Good training images
        good_imgs = self.get_train_good_images()
        for g in good_imgs:
            items.append((g, 0, "good"))

        # 2. Allotted real defects
        for d in allotted_real_defects:
            items.append((d["img_path"], 1, f"real_{d['defect_type']}"))

        # 3. Synthetic defects (if provided)
        if synthetic_defects:
            for s in synthetic_defects:
                items.append((s, 1, synthetic_label_name))

        return items


def get_transforms(is_train: bool = True):
    """Returns PyTorch transforms for ResNet18."""
    if is_train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE_CLASSIFIER),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE_CLASSIFIER),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

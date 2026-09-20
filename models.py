"""
SynthQC Models & Evaluation Module
Implements:
1. ResNet18 binary classifier for defect detection.
2. Arm A / B / C multi-run training engine (evaluating across 3-5 random splits).
3. Objective metrics computation (mean +/- std for Precision, Recall, F1, ROC-AUC).
4. Grad-CAM explainability localization hook.
5. Sanity check gate verifying Arm A achieves non-trivial performance (>70%).
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from config import (
    DEVICE, MODELS_DIR, IMAGE_SIZE_CLASSIFIER, RANDOM_SEED
)
from data_manager import DataManager, SynthQCDataset, get_transforms
from synthetic_generator import ProceduralGenerator, PatchGANTrainer


# ==============================================================================
# Model Architecture
# ==============================================================================

def create_model(num_classes: int = 2) -> nn.Module:
    """
    Constructs a ResNet18 classifier with pretrained weights.
    Sets up fine-tuning on layer4 and final fc layer for rapid CPU convergence.
    """
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    # Freeze earlier feature extractor layers for fast CPU training
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 and replace fc
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)


# ==============================================================================
# Grad-CAM Explainability Hook
# ==============================================================================

class GradCAM:
    """
    Grad-CAM implementation hooking into ResNet18 layer4 to produce
    visual localization heatmaps of model attention.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.layer4[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        Generates normalized Grad-CAM heatmap for the given input tensor.
        input_tensor shape: (1, 3, H, W)
        Returns: 2D numpy array of shape (H, W) in range [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max - cam_min > 1e-6:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize to input tensor dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam_resized

    def overlay_heatmap(self, original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlays heatmap on original RGB image using Jet colormap.
        original_rgb: (H, W, 3) in [0, 255]
        heatmap: (H, W) in [0, 1]
        Returns: (H, W, 3) blended RGB image
        """
        h, w, _ = original_rgb.shape
        heatmap_scaled = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_scaled * 255.0).astype(np.uint8)

        color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

        blended = (original_rgb.astype(np.float32) * (1.0 - alpha) + color_map_rgb.astype(np.float32) * alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)


# ==============================================================================
# Single Run Trainer
# ==============================================================================

def train_single_run(
    train_items: List[Tuple[str, int, str]],
    test_items: List[Tuple[str, int, str]],
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-4,
    progress_callback: Optional[Callable] = None
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Trains one ResNet18 model for a specific split and evaluates on the fixed held-out test set.
    """
    train_dataset = SynthQCDataset(train_items, transform=get_transforms(is_train=True))
    test_dataset = SynthQCDataset(test_items, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = create_model(num_classes=2)

    # Balanced class weights
    labels = [item[1] for item in train_items]
    good_count = max(1, labels.count(0))
    defect_count = max(1, labels.count(1))
    total = good_count + defect_count
    w0 = total / (2.0 * good_count)
    w1 = total / (2.0 * defect_count)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, lbls, _, _ in train_loader:
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if progress_callback:
            progress_callback(epoch + 1, epochs, epoch_loss / len(train_loader))

    # Evaluate on held-out test set
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for imgs, lbls, _, _ in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(lbls.numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Compute metrics
    prec = float(precision_score(all_targets, all_preds, zero_division=0))
    rec = float(recall_score(all_targets, all_preds, zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, zero_division=0))
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = (rec + spec) / 2.0

    try:
        auc = float(roc_auc_score(all_targets, all_probs))
    except Exception:
        auc = 0.5

    metrics = {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist()
    }

    return model, metrics


# ==============================================================================
# Multi-Split Evaluator (reporting mean +/- std)
# ==============================================================================

class ExperimentRunner:
    """
    Orchestrates scientific multi-split training for Arms A, B, and C.
    Guarantees:
    - Zero source leakage into synthetic generators.
    - Evaluation strictly on quarantined held-out test set.
    - Aggregation into mean +/- std across runs.
    """
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.fixed_test_items = self.dm.get_fixed_test_items()
        self.good_images = self.dm.get_train_good_images()

    def run_arm(
        self,
        arm_name: str,              # 'Arm A (Real Only)', 'Arm B (Real + Procedural)', 'Arm C (Real + GAN)'
        fraction: float,            # 0.10, 0.25, 0.50, 1.0
        num_splits: int = 3,
        epochs_per_run: int = 5,
        synthetic_count: int = 25,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Executes num_splits separate runs with independent seeds and aggregates results.
        """
        base_seeds = [42 + i * 17 for i in range(num_splits)]
        run_metrics = []
        trained_models = []
        gan_diagnostics = []

        for split_idx, seed in enumerate(base_seeds):
            # 1. Strictly sample allotted defects for this split
            allotted_defects = self.dm.sample_training_defects(fraction, split_seed=seed)

            synth_paths = []
            if "Arm B" in arm_name:
                # Procedural generator uses ONLY allotted defects
                proc_gen = ProceduralGenerator(allotted_defects, self.good_images)
                split_synth_dir = self.dm.synthetic_dir / f"run_proc_{seed}"
                synth_paths = proc_gen.generate_batch(
                    count=synthetic_count,
                    output_dir=split_synth_dir,
                    prefix=f"armb_s{seed}",
                    rng_seed=seed
                )
            elif "Arm C" in arm_name:
                # GAN uses ONLY allotted defects
                gan_trainer = PatchGANTrainer(allotted_defects, self.good_images)
                gan_res = gan_trainer.train_quick(epochs=8, batch_size=4)
                gan_diagnostics.append(gan_res)

                if not gan_res["is_degenerate"]:
                    split_synth_dir = self.dm.synthetic_dir / f"run_gan_{seed}"
                    split_synth_dir.mkdir(parents=True, exist_ok=True)
                    rng = random.Random(seed)
                    for i in range(synthetic_count):
                        g_img = rng.choice(self.good_images)
                        s_img, s_mask = gan_trainer.generate_full_synthetic_image(g_img, rng=rng)
                        if s_img is not None:
                            p = split_synth_dir / f"armc_s{seed}_{i:03d}.png"
                            cv2.imwrite(str(p), cv2.cvtColor(s_img, cv2.COLOR_RGB2BGR))
                            synth_paths.append(str(p))

            # 2. Build training set for this run
            train_items = self.dm.build_run_dataset(
                allotted_real_defects=allotted_defects,
                synthetic_defects=synth_paths if synth_paths else None,
                synthetic_label_name="synthetic"
            )

            # 3. Train ResNet18 and evaluate on quarantined held-out test set
            def split_cb(ep, total_ep, loss):
                if progress_callback:
                    overall_p = (split_idx + ep / total_ep) / num_splits
                    progress_callback(overall_p, f"{arm_name} (frac {int(fraction*100)}%) Split {split_idx+1}/{num_splits} | Ep {ep}/{total_ep} Loss: {loss:.3f}")

            model, metrics = train_single_run(
                train_items=train_items,
                test_items=self.fixed_test_items,
                epochs=epochs_per_run,
                progress_callback=split_cb
            )

            run_metrics.append(metrics)
            trained_models.append(model)

        # Aggregate across splits: mean +/- std
        agg_results = self._aggregate_metrics(run_metrics)
        agg_results["arm_name"] = arm_name
        agg_results["fraction"] = fraction
        agg_results["allotted_defect_count"] = len(allotted_defects)
        agg_results["num_splits"] = num_splits
        agg_results["raw_runs"] = run_metrics
        agg_results["best_model"] = trained_models[0]  # Store first model for live inspection
        agg_results["gan_diagnostics"] = gan_diagnostics

        return agg_results

    def _aggregate_metrics(self, runs: List[Dict]) -> Dict:
        """Calculates mean and standard deviation for each metric."""
        metric_keys = ["precision", "recall", "f1", "specificity", "balanced_accuracy", "roc_auc"]
        aggregated = {}

        for k in metric_keys:
            vals = [r[k] for r in runs]
            aggregated[f"{k}_mean"] = float(np.mean(vals))
            aggregated[f"{k}_std"] = float(np.std(vals))

        return aggregated


# ==============================================================================
# Sanity Check Gate
# ==============================================================================

def run_sanity_check(experiment_runner: ExperimentRunner) -> Dict:
    """
    Verifies that Arm A (real defects only, 100% pool) achieves non-trivial
    performance on the held-out test set (>70% balanced accuracy).
    Treats chance performance as a configuration error that must be resolved.
    """
    print("[Sanity Check] Evaluating Arm A on 100% pool across 3 splits...")
    res = experiment_runner.run_arm(
        arm_name="Arm A (Real Only)",
        fraction=1.0,
        num_splits=3,
        epochs_per_run=5
    )

    bal_acc = res["balanced_accuracy_mean"]
    f1 = res["f1_mean"]
    passed = bal_acc >= 0.70 and f1 >= 0.70

    return {
        "passed": passed,
        "balanced_accuracy_mean": bal_acc,
        "f1_mean": f1,
        "result_details": res
    }

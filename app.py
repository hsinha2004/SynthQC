"""
SynthQC - Optical Quality Inspection & Defect Synthesis Platform
Architecture:
1. Dataset & Partition Telemetry
2. Defect Synthesis Studio (Procedural Poisson & Patch cGAN)
3. Model Training & Data Scarcity Benchmarks
4. Optical Inspection (Grad-CAM Explainability & Bounding Box IoU)
5. Benchmarking & Decision Threshold Analysis
"""

import os
import sys
import time
import json
import random
import io
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt

import streamlit as st

# Setup page configuration
st.set_page_config(
    page_title="SynthQC - Optical Inspection System",
    page_icon="⊙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import backend modules
from config import (
    PRODUCT_CATEGORY, DEVICE, IMAGE_SIZE_CLASSIFIER,
    THEME_PRIMARY, THEME_SUCCESS, THEME_WARNING, THEME_DANGER, THEME_BG, THEME_CARD,
    THEME_BORDER, THEME_TEXT_PRIMARY, THEME_TEXT_SECONDARY, THEME_TEXT_MUTED
)
from data_manager import DataManager, get_transforms
from synthetic_generator import (
    ProceduralGenerator, PatchGANTrainer, extract_defect_patches,
    inject_interactive_sample
)
from models import (
    create_model, GradCAM, ExperimentRunner, run_sanity_check,
    train_single_run, evaluate_at_threshold, batch_inspect_images
)

# Initialize Session State
if "data_manager" not in st.session_state:
    st.session_state.data_manager = DataManager()

if "experiment_runner" not in st.session_state:
    st.session_state.experiment_runner = ExperimentRunner(st.session_state.data_manager)

if "benchmark_results" not in st.session_state:
    cache_path = Path("data/benchmark_cache.json")
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                st.session_state.benchmark_results = json.load(f)
        except Exception:
            st.session_state.benchmark_results = []
    else:
        st.session_state.benchmark_results = []

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
    model_path = Path("saved_models/arm_a_model.pt")
    if model_path.exists():
        try:
            m = create_model()
            m.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            m.eval()
            st.session_state.trained_models["Arm A (Real Only) [100%]"] = m
        except Exception as e:
            print(f"Could not load pre-trained model: {e}")

if "inspection_log" not in st.session_state:
    st.session_state.inspection_log = []

if "active_grad_cam" not in st.session_state:
    st.session_state.active_grad_cam = None

if "sanity_check_status" not in st.session_state:
    st.session_state.sanity_check_status = None

if "operating_threshold" not in st.session_state:
    st.session_state.operating_threshold = 0.50

dm = st.session_state.data_manager
runner = st.session_state.experiment_runner

# ==============================================================================
# CSS & Styling (Unslopped: Calibrated Industrial Metrology Design)
# ==============================================================================

st.markdown(f"""
<style>
    /* Base typography */
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .main-header {{
        font-size: 1.65rem;
        font-weight: 600;
        letter-spacing: -0.3px;
        color: {THEME_TEXT_PRIMARY};
        margin-bottom: 0.2rem;
    }}
    .sub-header {{
        color: {THEME_TEXT_SECONDARY};
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
        line-height: 1.4;
    }}
    /* Quiet industrial telemetry cards */
    .telemetry-card {{
        background: {THEME_CARD};
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid {THEME_BORDER};
        text-align: left;
    }}
    .telemetry-val {{
        font-size: 1.5rem;
        font-weight: 600;
        font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
        color: {THEME_TEXT_PRIMARY};
        letter-spacing: -0.5px;
    }}
    .telemetry-lbl {{
        font-size: 0.75rem;
        color: {THEME_TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 0.2rem;
    }}
    /* Precise status indicators */
    .telemetry-tag {{
        display: inline-block;
        font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
    }}
    .tag-reject {{
        background: rgba(211, 63, 73, 0.12);
        color: {THEME_DANGER};
        border: 1px solid rgba(211, 63, 73, 0.35);
    }}
    .tag-pass {{
        background: rgba(46, 158, 102, 0.12);
        color: {THEME_SUCCESS};
        border: 1px solid rgba(46, 158, 102, 0.35);
    }}
    .tag-neutral {{
        background: rgba(255, 255, 255, 0.05);
        color: {THEME_TEXT_SECONDARY};
        border: 1px solid {THEME_BORDER};
    }}
    .tag-accent {{
        background: rgba(74, 114, 176, 0.14);
        color: #7AA2E3;
        border: 1px solid rgba(74, 114, 176, 0.35);
    }}
    /* Quiet borders and dividers */
    hr {{
        border: none;
        border-top: 1px solid {THEME_BORDER};
        margin: 1.5rem 0;
    }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Sidebar Navigation & Telemetry
# ==============================================================================

st.sidebar.markdown(f"### SynthQC / Vision Engine")
st.sidebar.caption("Optical Metrology & Synthetic Defect Benchmark")
st.sidebar.markdown("---")

st.sidebar.markdown("**System Telemetry**")
st.sidebar.write(f"Part Profile: `{PRODUCT_CATEGORY}`")
st.sidebar.write(f"Compute Engine: `{DEVICE}`")
cgan_ready = Path("saved_models/patch_cgan_gen.pt").exists()
st.sidebar.markdown(f"Generator Model: <span class='telemetry-tag {'tag-accent' if cgan_ready else 'tag-neutral'}'>{'cGAN Checkpoint Active' if cgan_ready else 'Uninitialized'}</span>", unsafe_allow_html=True)
st.sidebar.write("Dataset License: Non-commercial academic (CC BY-NC-SA 4.0)")

counts = dm.get_subtype_counts()
held_out = counts.get("held_out_test", {})
pool = counts.get("training_pool", {})

st.sidebar.markdown("---")
st.sidebar.markdown("**Partition Telemetry**")
st.sidebar.write(f"Nominal Reference: {counts.get('train_good', 0)} samples")
st.sidebar.write(f"Quarantined Test Set: {sum(v for k, v in held_out.items() if k != 'good')} anomalies | {held_out.get('good', 0)} nominal")
st.sidebar.write(f"Training Pool Pool: {sum(pool.values())} anomalies")

st.sidebar.markdown("---")
st.sidebar.markdown("[Repository & Source Code](https://github.com/hsinha2004/SynthQC)")

tabs = st.tabs([
    "Dataset",
    "Defect Synthesis",
    "Model Training",
    "Optical Inspection",
    "Benchmarking & ROC"
])


# ==============================================================================
# TAB 1: DATASET VIEW
# ==============================================================================
with tabs[0]:
    st.markdown('<div class="main-header">Dataset Partitions & Ground-Truth Annotation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Review image distributions, defect subtypes, and quarantined validation sets.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val">{counts.get("train_good", 0)}</div><div class="telemetry-lbl">Nominal Training Set</div></div>', unsafe_allow_html=True)
    with col2:
        test_defect_sum = sum(v for k, v in held_out.items() if k != 'good')
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val">{test_defect_sum}</div><div class="telemetry-lbl">Quarantined Test Anomalies</div></div>', unsafe_allow_html=True)
    with col3:
        pool_defect_sum = sum(pool.values())
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val">{pool_defect_sum}</div><div class="telemetry-lbl">Allottable Training Defects</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val">{held_out.get("good", 0)}</div><div class="telemetry-lbl">Quarantined Test Nominal</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Subtype Stratification Breakdown**")
    subtypes_df = pd.DataFrame([
        {"Defect Class": "broken_large", "Quarantined Test Split": held_out.get("broken_large", 0), "Allottable Pool": pool.get("broken_large", 0), "Total Available": held_out.get("broken_large", 0) + pool.get("broken_large", 0)},
        {"Defect Class": "broken_small", "Quarantined Test Split": held_out.get("broken_small", 0), "Allottable Pool": pool.get("broken_small", 0), "Total Available": held_out.get("broken_small", 0) + pool.get("broken_small", 0)},
        {"Defect Class": "contamination", "Quarantined Test Split": held_out.get("contamination", 0), "Allottable Pool": pool.get("contamination", 0), "Total Available": held_out.get("contamination", 0) + pool.get("contamination", 0)},
        {"Defect Class": "nominal (good)", "Quarantined Test Split": held_out.get("good", 0), "Allottable Pool": counts.get("train_good", 0), "Total Available": held_out.get("good", 0) + counts.get("train_good", 0)},
    ])
    st.dataframe(subtypes_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("**Sample Inspection & Annotation Viewer**")

    browse_partition = st.radio(
        "Select Partition Scope:",
        ["Allottable Training Pool", "Quarantined Held-Out Test Set", "Nominal Reference Set"],
        horizontal=True
    )

    if browse_partition == "Nominal Reference Set":
        good_imgs = dm.get_train_good_images()
        if good_imgs:
            idx = st.slider("Index:", 0, len(good_imgs) - 1, 0)
            img_p = good_imgs[idx]
            st.image(img_p, caption=f"Nominal Sample: {Path(img_p).name}", width=400)
    else:
        items_to_browse = dm.split_registry.get("training_pool" if "Allottable" in browse_partition else "held_out_test", [])
        if items_to_browse:
            subtypes_avail = sorted(list(set(it["defect_type"] for it in items_to_browse)))
            chosen_sub = st.selectbox("Defect Subtype:", subtypes_avail)
            filtered = [it for it in items_to_browse if it["defect_type"] == chosen_sub]

            if filtered:
                sel_idx = st.slider(f"Sample Index ({chosen_sub}):", 0, len(filtered) - 1, 0)
                item = filtered[sel_idx]

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    st.caption(f"Inspection Capture: `{item.get('stem')}.png`")
                    if os.path.exists(item["img_path"]):
                        st.image(item["img_path"], use_container_width=True)
                with b_col2:
                    st.caption("Binary Ground-Truth Mask")
                    if item.get("mask_path") and os.path.exists(item["mask_path"]):
                        st.image(item["mask_path"], use_container_width=True)
                    else:
                        st.info("No anomaly mask associated with nominal sample.")


# ==============================================================================
# TAB 2: SYNTHETIC DEFECT STUDIO
# ==============================================================================
with tabs[1]:
    st.markdown('<div class="main-header">Defect Synthesis Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Synthesize defect profiles via procedural Poisson cloning and 64x64 conditional Patch-GAN.</div>', unsafe_allow_html=True)

    studio_mode = st.radio(
        "Mode:",
        ["Interactive Studio", "Batch Synthesis", "Method Comparison"],
        horizontal=True
    )

    good_imgs = dm.get_train_good_images()

    if studio_mode == "Interactive Studio":
        s_col1, s_col2 = st.columns([1, 2])

        with s_col1:
            st.markdown("**Synthesis Parameters**")
            studio_frac = st.select_slider(
                "Source Defect Pool Fraction:",
                options=[0.10, 0.25, 0.50, 1.0],
                value=0.25,
                format_func=lambda x: f"{int(x*100)}% ({max(1, int(round(39*x)))} source images)"
            )
            allotted_defects = dm.sample_training_defects(studio_frac, 42)
            st.caption(f"Zero-leakage: isolated from {len(allotted_defects)} allotted training samples.")

            proc_gen = ProceduralGenerator(allotted_defects, good_imgs)

            chosen_blend = st.selectbox("Blending Formulation:", ["poisson_normal", "poisson_mixed", "alpha"])
            pos_x = st.slider("Center X:", 120, 380, 250, step=5)
            pos_y = st.slider("Center Y:", 120, 380, 250, step=5)
            scale_val = st.slider("Relative Scale:", 0.5, 1.6, 1.0, step=0.05)
            rot_val = st.slider("Rotation Angle (deg):", -180, 180, 0, step=5)
            feather_val = st.slider("Feather Radius (px):", 3, 25, 11, step=2)

            patch_idx = st.number_input("Source Patch Index:", min_value=0, max_value=max(0, len(proc_gen.patches) - 1), value=0)
            good_idx = st.number_input("Base Nominal Index:", min_value=0, max_value=max(0, len(good_imgs) - 1), value=0)

        with s_col2:
            st.markdown("**Synthesis Preview**")
            synth_rgb, synth_mask, dtype, meta = proc_gen.generate_single(
                blend_mode=chosen_blend,
                manual_coords=(pos_x, pos_y),
                patch_idx=patch_idx,
                good_img_idx=good_idx,
                explicit_scale=scale_val,
                explicit_rotation=rot_val,
                feather_radius=feather_val
            )

            if synth_rgb is not None:
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    st.image(meta["good_rgb"], caption="Base Nominal", use_container_width=True)
                with p2:
                    st.image(meta["patch_img"], caption=f"Source Patch ({dtype})", use_container_width=True)
                with p3:
                    st.image(synth_rgb, caption="Synthetic Defect", use_container_width=True)
                with p4:
                    st.image(synth_mask, caption="Synthesized Mask", use_container_width=True)

                st.markdown("---")
                if st.button("Commit to Training Pool"):
                    p_img, p_mask = inject_interactive_sample(synth_rgb, synth_mask, prefix="studio_sample")
                    st.success(f"Sample committed: `{Path(p_img).name}`")

    elif studio_mode == "Batch Synthesis":
        b_col1, b_col2 = st.columns([1, 2])
        with b_col1:
            st.markdown("**Batch Synthesis Configuration**")
            b_frac = st.select_slider("Allotted Pool Fraction:", options=[0.10, 0.25, 0.50, 1.0], value=0.50)
            b_count = st.slider("Target Batch Size:", 5, 50, 20)
            b_blend = st.selectbox("Blending Formulation:", ["poisson_normal", "poisson_mixed", "alpha"])
            b_seed = st.number_input("Random Seed:", value=42, step=1)
            b_gen_btn = st.button("Generate Batch")

        with b_col2:
            st.markdown("**Batch Output**")
            if b_gen_btn:
                allotted_defects = dm.sample_training_defects(b_frac, b_seed)
                proc_gen = ProceduralGenerator(allotted_defects, good_imgs)
                out_dir = dm.synthetic_dir / f"batch_gen_{b_seed}"

                with st.spinner(f"Synthesizing {b_count} samples..."):
                    paths = proc_gen.generate_batch(b_count, out_dir, prefix="batch_synth", rng_seed=b_seed)

                st.success(f"Completed {len(paths)} synthesized defect images in `{out_dir.name}`")

                preview_paths = paths[:6]
                g_cols = st.columns(min(len(preview_paths), 3))
                for i, p in enumerate(preview_paths):
                    col = g_cols[i % len(g_cols)]
                    col.image(p, caption=Path(p).name, use_container_width=True)

    elif studio_mode == "Method Comparison":
        st.markdown("**Synthesis Formulations: Real vs. Procedural Poisson vs. 64x64 Patch cGAN**")
        st.caption("Visual inspection of artifact characteristics across defect generation strategies.")

        allotted_defects = dm.sample_training_defects(1.0, 42)
        proc_gen = ProceduralGenerator(allotted_defects, good_imgs)
        gan_trainer = PatchGANTrainer(allotted_defects, good_imgs)
        cgan_loaded = gan_trainer.load_pretrained("saved_models/patch_cgan_gen.pt")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Authentic Real Defect**")
            if allotted_defects:
                sample_real = allotted_defects[0]
                st.image(sample_real["img_path"], caption=f"Class: {sample_real['defect_type']}", use_container_width=True)
                if sample_real.get("mask_path") and os.path.exists(sample_real["mask_path"]):
                    st.image(sample_real["mask_path"], caption="Ground-Truth Annotation", use_container_width=True)

        with c2:
            st.markdown("**2. Procedural Poisson Blend**")
            s_proc, s_mask_proc, p_type, _ = proc_gen.generate_single(blend_mode="poisson_normal")
            if s_proc is not None:
                st.image(s_proc, caption=f"Synthesized: {p_type}", use_container_width=True)
                st.image(s_mask_proc, caption="Procedural Mask", use_container_width=True)

        with c3:
            st.markdown("**3. Patch cGAN Composite**")
            if cgan_loaded:
                s_gan, s_mask_gan, gan_meta = gan_trainer.generate_full_synthetic_image(good_imgs[0])
                if s_gan is not None:
                    st.image(s_gan, caption="cGAN Generated Patch", use_container_width=True)
                    st.image(s_mask_gan, caption="Synthesized Mask", use_container_width=True)
                else:
                    st.info("Composite generation failed.")
            else:
                st.info("Checkpoint `saved_models/patch_cgan_gen.pt` not located.")


# ==============================================================================
# TAB 3: TRAINING VIEW
# ==============================================================================
with tabs[2]:
    st.markdown('<div class="main-header">Model Training & Scarcity Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Train ResNet18 backbones across experimental arms and compute multi-split metrics.</div>', unsafe_allow_html=True)

    t_col1, t_col2 = st.columns([1, 2])

    with t_col1:
        st.markdown("**Experiment Setup**")

        train_frac = st.select_slider(
            "Defect Scarcity Fraction:",
            options=[0.10, 0.25, 0.50, 1.0],
            value=0.25,
            format_func=lambda x: f"{int(x*100)}% ({max(1, int(round(39*x)))} source samples)"
        )

        num_splits = st.slider("Cross-Validation Splits (Seeds):", min_value=2, max_value=5, value=3)
        epochs_per_run = st.slider("Epochs per Split:", min_value=3, max_value=10, value=5)

        st.markdown("---")
        st.markdown("**Model Arms**")
        train_arm_a = st.checkbox("Arm A (Real Defects Only)", value=True)
        train_arm_b = st.checkbox("Arm B (Real + Procedural Poisson)", value=True)
        train_arm_c = st.checkbox("Arm C (Real + Patch-GAN)", value=False)

        st.markdown("---")
        sanity_btn = st.button("Verify Arm A Baseline")
        train_btn = st.button("Execute Training Run")

    with t_col2:
        st.markdown("**Training Telemetry & Convergence**")

        if sanity_btn:
            with st.spinner("Evaluating baseline performance across 3 independent splits..."):
                sanity_res = run_sanity_check(runner)
                st.session_state.sanity_check_status = sanity_res

        if st.session_state.sanity_check_status:
            sc = st.session_state.sanity_check_status
            if sc["passed"]:
                st.markdown(f"<span class='telemetry-tag tag-pass'>BASELINE PASSED</span> Balanced Acc: <b>{sc['balanced_accuracy_mean']*100:.1f}%</b> | F1: <b>{sc['f1_mean']*100:.1f}%</b>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='telemetry-tag tag-reject'>BASELINE BELOW THRESHOLD</span> Balanced Acc: <b>{sc['balanced_accuracy_mean']*100:.1f}%</b>", unsafe_allow_html=True)

        if train_btn:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            arms_to_run = []
            if train_arm_a:
                arms_to_run.append("Arm A (Real Only)")
            if train_arm_b:
                arms_to_run.append("Arm B (Real + Procedural)")
            if train_arm_c:
                arms_to_run.append("Arm C (Real + GAN)")

            if not arms_to_run:
                st.warning("Select at least one experimental arm.")
            else:
                for a_idx, arm_name in enumerate(arms_to_run):
                    status_text.write(f"Executing {arm_name} across {num_splits} splits...")

                    def cb(p, msg):
                        progress_bar.progress((a_idx + p) / len(arms_to_run))
                        status_text.text(msg)

                    res = runner.run_arm(
                        arm_name=arm_name,
                        fraction=train_frac,
                        num_splits=num_splits,
                        epochs_per_run=epochs_per_run,
                        progress_callback=cb
                    )

                    st.session_state.benchmark_results.append(res)
                    st.session_state.trained_models[f"{arm_name} [{int(train_frac*100)}%]"] = res["best_model"]

                progress_bar.progress(1.0)
                status_text.text("Training complete. Evaluated against held-out split.")

        if st.session_state.benchmark_results:
            st.markdown("**Recent Benchmark Runs**")
            display_rows = []
            for r in st.session_state.benchmark_results[-8:]:
                display_rows.append({
                    "Arm": r["arm_name"],
                    "Scarcity": f"{int(r['fraction']*100)}%",
                    "Real Count": r["allotted_defect_count"],
                    "Precision": f"{r['precision_mean']*100:.1f} ± {r['precision_std']*100:.1f}%",
                    "Recall": f"{r['recall_mean']*100:.1f} ± {r['recall_std']*100:.1f}%",
                    "F1 Score": f"{r['f1_mean']*100:.1f} ± {r['f1_std']*100:.1f}%",
                    "Balanced Acc": f"{r['balanced_accuracy_mean']*100:.1f} ± {r['balanced_accuracy_std']*100:.1f}%",
                    "ROC-AUC": f"{r['roc_auc_mean']:.3f} ± {r['roc_auc_std']:.3f}",
                })
            st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)


# ==============================================================================
# TAB 4: OPTICAL INSPECTION & QA
# ==============================================================================
with tabs[3]:
    st.markdown('<div class="main-header">Optical Inspection & Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate test specimens, inspect Grad-CAM localization heatmaps, and record quality dispositions.</div>', unsafe_allow_html=True)

    inspect_mode = st.radio(
        "Inspection Scope:",
        ["Single Part Detailed Inspection", "Batch Inspection Grid"],
        horizontal=True
    )

    avail_models = list(st.session_state.trained_models.keys())
    if not avail_models:
        st.info("No custom models in session. Utilizing ResNet18 baseline checkpoint.")
        selected_model_name = "ResNet18 Baseline"
        active_model = create_model()
    else:
        selected_model_name = st.selectbox("Active Inference Model:", avail_models)
        active_model = st.session_state.trained_models[selected_model_name]

    test_items = dm.get_fixed_test_items()

    if inspect_mode == "Single Part Detailed Inspection":
        ins_col1, ins_col2 = st.columns([1, 2])

        with ins_col1:
            st.markdown("**Part Acquisition**")
            img_source = st.radio("Source:", ["Quarantined Held-Out Test Set", "External Upload"])

            selected_img_path = None
            ground_truth_label = None
            mask_path_for_iou = None

            if img_source == "Quarantined Held-Out Test Set":
                test_subtypes = sorted(list(set(it[2] for it in test_items)))
                sel_st = st.selectbox("Filter Class:", test_subtypes)
                matching = [it for it in test_items if it[2] == sel_st]
                sel_idx = st.slider("Sample Index:", 0, len(matching) - 1, 0)
                selected_img_path = matching[sel_idx][0]
                ground_truth_label = "Defective" if matching[sel_idx][1] == 1 else "Nominal (Good)"

                if matching[sel_idx][1] == 1:
                    stem = Path(selected_img_path).stem
                    possible_mask = dm.gt_dir / sel_st / f"{stem}_mask.png"
                    if possible_mask.exists():
                        mask_path_for_iou = str(possible_mask)
            else:
                uploaded = st.file_uploader("Upload Image:", type=["png", "jpg", "jpeg"])
                if uploaded:
                    save_path = dm.category_dir / f"custom_upload_{uploaded.name}"
                    with open(save_path, "wb") as f:
                        f.write(uploaded.read())
                    selected_img_path = str(save_path)
                    ground_truth_label = "Unverified"

            run_inspect_btn = st.button("Inspect Part")

        with ins_col2:
            st.markdown("**Optical Telemetry & Localization**")

            if selected_img_path and os.path.exists(selected_img_path):
                img_pil = Image.open(selected_img_path).convert("RGB")
                orig_rgb = np.array(img_pil)

                t_img = get_transforms(is_train=False)(img_pil).unsqueeze(0).to(DEVICE)
                active_model.eval()

                with torch.no_grad():
                    output = active_model(t_img)
                    probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                    pred_class = int(np.argmax(probs))
                    defect_prob = float(probs[1])

                grad_cam = GradCAM(active_model)
                cam = grad_cam.generate_heatmap(t_img, class_idx=1)
                overlay = grad_cam.overlay_heatmap(orig_rgb, cam, alpha=0.55)
                iou_res = grad_cam.compute_iou_against_mask(cam, mask_path_for_iou)

                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    if pred_class == 1 or defect_prob >= st.session_state.operating_threshold:
                        st.markdown(f'<span class="telemetry-tag tag-reject">ANOMALY DETECTED [{defect_prob*100:.1f}%]</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="telemetry-tag tag-pass">NOMINAL [{(1.0-defect_prob)*100:.1f}%]</span>', unsafe_allow_html=True)

                with r_col2:
                    st.write(f"Ground Truth: `{ground_truth_label}`")
                    if iou_res.get("has_gt") and iou_res.get("iou") is not None:
                        st.markdown(f"Grad-CAM Localization IoU: <span class='telemetry-tag tag-accent'>{iou_res['iou']*100:.1f}%</span> (Dice: {iou_res['dice']*100:.1f}%)", unsafe_allow_html=True)

                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.image(orig_rgb, caption="Raw Optical Feed", use_container_width=True)
                with v_col2:
                    st.image(overlay, caption="Grad-CAM Activation Overlay", use_container_width=True)

                st.markdown("---")
                st.markdown("**Operator Quality Disposition**")
                qa_col1, qa_col2, qa_col3 = st.columns(3)
                with qa_col1:
                    if st.button("Accept AI Clearance", use_container_width=True):
                        st.session_state.inspection_log.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image": Path(selected_img_path).name,
                            "model": selected_model_name,
                            "defect_prob": f"{defect_prob*100:.1f}%",
                            "ai_decision": "Defect" if defect_prob >= st.session_state.operating_threshold else "Good",
                            "disposition": "Accepted",
                            "ground_truth": ground_truth_label,
                            "iou": f"{iou_res['iou']*100:.1f}%" if iou_res.get("has_gt") else "N/A"
                        })
                        st.success("Disposition logged: Accepted")
                with qa_col2:
                    if st.button("Override Disposition", use_container_width=True):
                        st.session_state.inspection_log.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image": Path(selected_img_path).name,
                            "model": selected_model_name,
                            "defect_prob": f"{defect_prob*100:.1f}%",
                            "ai_decision": "Defect" if defect_prob >= st.session_state.operating_threshold else "Good",
                            "disposition": "Overridden",
                            "ground_truth": ground_truth_label,
                            "iou": f"{iou_res['iou']*100:.1f}%" if iou_res.get("has_gt") else "N/A"
                        })
                        st.warning("Disposition logged: Overridden")
                with qa_col3:
                    if st.button("Route to Secondary Audit", use_container_width=True):
                        st.session_state.inspection_log.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image": Path(selected_img_path).name,
                            "model": selected_model_name,
                            "defect_prob": f"{defect_prob*100:.1f}%",
                            "ai_decision": "Defect" if defect_prob >= st.session_state.operating_threshold else "Good",
                            "disposition": "Secondary Audit",
                            "ground_truth": ground_truth_label,
                            "iou": f"{iou_res['iou']*100:.1f}%" if iou_res.get("has_gt") else "N/A"
                        })
                        st.info("Part routed to secondary audit")

    elif inspect_mode == "Batch Inspection Grid":
        st.markdown("**Batch Quality Screening**")
        st.caption("Parallel optical inference across the quarantined held-out validation set.")

        batch_items = []
        for it in test_items:
            img_p, lbl, dt = it
            stem = Path(img_p).stem
            mask_p = None
            if lbl == 1:
                p = dm.gt_dir / dt / f"{stem}_mask.png"
                if p.exists():
                    mask_p = str(p)
            batch_items.append({
                "img_path": img_p,
                "label": lbl,
                "defect_type": dt,
                "stem": stem,
                "mask_path": mask_p
            })

        batch_limit = st.slider("Screening Sample Count:", 4, len(batch_items), 12, step=4)
        run_batch_btn = st.button("Execute Batch Screening")

        if run_batch_btn:
            with st.spinner("Executing screening and activation mapping..."):
                batch_results = batch_inspect_images(active_model, batch_items[:batch_limit], threshold=st.session_state.operating_threshold)

            total = len(batch_results)
            flagged = sum(1 for r in batch_results if r["is_flagged"])
            ious = [r["iou"] for r in batch_results if r.get("iou") is not None]
            mean_iou = float(np.mean(ious)) if ious else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Screened Parts", total)
            m2.metric("Rejected (Anomalies)", flagged)
            m3.metric("Cleared (Nominal)", total - flagged)
            m4.metric("Mean Localization IoU", f"{mean_iou*100:.1f}%")

            st.markdown("---")
            cols_per_row = 3
            for r_i in range(0, len(batch_results), cols_per_row):
                row_items = batch_results[r_i:r_i + cols_per_row]
                row_cols = st.columns(cols_per_row)
                for c_i, r in enumerate(row_items):
                    with row_cols[c_i]:
                        status_str = "REJECT" if r["is_flagged"] else "CLEAR"
                        iou_disp = f"{r['iou']*100:.1f}%" if r.get("iou") is not None else "N/A"
                        st.image(r["overlay_rgb"], caption=f"{r['stem']} | {status_str} ({r['defect_prob']*100:.1f}%)", use_container_width=True)
                        st.caption(f"Ground Truth: `{r['defect_type']}` | IoU: `{iou_disp}`")


# ==============================================================================
# TAB 5: BENCHMARKING & ROC ANALYSIS
# ==============================================================================
with tabs[4]:
    st.markdown('<div class="main-header">Benchmarking & Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical data-efficiency comparisons and operating threshold tuning on the held-out validation set.</div>', unsafe_allow_html=True)

    st.markdown("**Operating Threshold Calibration**")
    st.caption("Adjust decision threshold T to evaluate the trade-off between False Alarms and Missed Defects.")

    t_thresh = st.slider(
        "Decision Threshold T (P(Defect) >= T triggers rejection):",
        min_value=0.05, max_value=0.95, value=float(st.session_state.operating_threshold), step=0.05
    )
    st.session_state.operating_threshold = t_thresh

    if avail_models:
        sel_m = st.session_state.trained_models.get(list(st.session_state.trained_models.keys())[0], None)
        if sel_m is not None:
            t_eval = evaluate_at_threshold(sel_m, test_items, threshold=t_thresh)

            tm1, tm2, tm3, tm4 = st.columns(4)
            tm1.metric("Precision", f"{t_eval['precision']*100:.1f}%")
            tm2.metric("Recall (Sensitivity)", f"{t_eval['recall']*100:.1f}%")
            tm3.metric("Specificity", f"{t_eval['specificity']*100:.1f}%")
            tm4.metric("False Alarm Rate", f"{t_eval['false_alarm_rate']*100:.1f}%")

            c_cm1, c_cm2 = st.columns([1, 1])
            with c_cm1:
                st.markdown(f"""
                | Confusion Matrix | Actual Nominal (0) | Actual Defective (1) |
                | :--- | :---: | :---: |
                | **AI Decision: Nominal** | **TN: {t_eval['tn']}** | FN (Missed): {t_eval['fn']} |
                | **AI Decision: Reject** | FP (False Alarm): {t_eval['fp']} | **TP: {t_eval['tp']}** |
                """)
            with c_cm2:
                fpr = t_eval['false_alarm_rate']
                tpr = t_eval['recall']
                fig_roc, ax_roc = plt.subplots(figsize=(6, 3), facecolor='#0D1117')
                ax_roc.set_facecolor('#161B22')
                ax_roc.plot([0, 1], [0, 1], '--', color='#6E7681', lw=1)
                ax_roc.plot([0, fpr, 1], [0, tpr, 1], color='#4A72B0', lw=2, label=f"AUC = {t_eval['roc_auc']:.3f}")
                ax_roc.plot([fpr], [tpr], 'o', color='#D33F49', markersize=8, label=f"T = {t_thresh:.2f}")
                ax_roc.set_xlabel('False Positive Rate', color='#8B949E', fontsize=9)
                ax_roc.set_ylabel('True Positive Rate', color='#8B949E', fontsize=9)
                ax_roc.set_title('Operating Point on ROC Space', color='#F0F6FC', fontsize=10)
                ax_roc.tick_params(colors='#8B949E', labelsize=8)
                ax_roc.grid(True, color='rgba(255,255,255,0.06)')
                ax_roc.legend(facecolor='#161B22', labelcolor='#F0F6FC', fontsize=8)
                st.pyplot(fig_roc)
                plt.close()

    st.markdown("---")

    if st.session_state.benchmark_results:
        st.markdown("**Cross-Arm Performance Metrics**")
        table_rows = []
        for r in st.session_state.benchmark_results:
            table_rows.append({
                "Arm": r["arm_name"],
                "Defect Pool Fraction": f"{int(r['fraction'] * 100)}%",
                "Source Count": r["allotted_defect_count"],
                "Precision": f"{r['precision_mean']*100:.1f} ± {r['precision_std']*100:.1f}%",
                "Recall": f"{r['recall_mean']*100:.1f} ± {r['recall_std']*100:.1f}%",
                "F1 Score": f"{r['f1_mean']*100:.1f} ± {r['f1_std']*100:.1f}%",
                "Balanced Accuracy": f"{r['balanced_accuracy_mean']*100:.1f} ± {r['balanced_accuracy_std']*100:.1f}%",
                "ROC-AUC": f"{r['roc_auc_mean']:.3f} ± {r['roc_auc_std']:.3f}",
            })

        df_bench = pd.DataFrame(table_rows)
        st.dataframe(df_bench, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("**Data Efficiency Trajectory**")
        st.caption("Quantifies inspection performance relative to real defect data scarcity.")

        plot_data = []
        for r in st.session_state.benchmark_results:
            plot_data.append({
                "Defect Pool Fraction (%)": int(r["fraction"] * 100),
                "F1 Score": r["f1_mean"] * 100.0,
                "Balanced Accuracy": r["balanced_accuracy_mean"] * 100.0,
                "Recall": r["recall_mean"] * 100.0,
                "Arm": r["arm_name"]
            })

        df_plot = pd.DataFrame(plot_data)
        metric_to_plot = st.selectbox("Trajectory Metric:", ["F1 Score", "Balanced Accuracy", "Recall"])

        pivot_df = df_plot.pivot_table(
            index="Defect Pool Fraction (%)",
            columns="Arm",
            values=metric_to_plot,
            aggfunc="mean"
        ).sort_index()

        st.line_chart(pivot_df)

    # Human Review Log & CSV Export
    st.markdown("---")
    st.markdown("**Quality Audit Log**")
    if st.session_state.inspection_log:
        df_log = pd.DataFrame(st.session_state.inspection_log)
        st.dataframe(df_log, hide_index=True, use_container_width=True)

        csv_buffer = io.StringIO()
        df_log.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Export Audit Log (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"synthqc_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.caption("No operator dispositions recorded in current session.")

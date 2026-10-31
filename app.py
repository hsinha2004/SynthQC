"""
SynthQC - AI-Assisted Visual Quality Inspection Web Application (Advanced Edition)
Features:
1. 📦 Dataset View & Partition Explorer
2. 🧪 Synthetic Defect Studio (Interactive Placement, Procedural Poisson & Patch cGAN)
3. ⚡ Multi-Arm Training (Defect Scarcity Simulation & Zero Leakage)
4. 🔍 Live Inspection & Batch QA (Grad-CAM Heatmaps, IoU Localization, Human QA Station)
5. 📊 Data Efficiency & Decision Threshold Dashboard (Dynamic Threshold Tuner & ROC Curve)
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
    page_title="SynthQC - Visual Quality Inspection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import backend modules
from config import (
    PRODUCT_CATEGORY, DEVICE, IMAGE_SIZE_CLASSIFIER,
    THEME_PRIMARY, THEME_SUCCESS, THEME_WARNING, THEME_DANGER, THEME_BG, THEME_CARD
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
# CSS & Styling
# ==============================================================================

st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }}
    .sub-header {{
        color: #8E9BB0;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }}
    .stat-card {{
        background: {THEME_CARD};
        padding: 1.1rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }}
    .stat-val {{
        font-size: 1.7rem;
        font-weight: 700;
        color: {THEME_PRIMARY};
    }}
    .stat-lbl {{
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .flag-defect {{
        background: rgba(255, 51, 102, 0.15);
        color: {THEME_DANGER};
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid {THEME_DANGER};
        display: inline-block;
    }}
    .flag-pass {{
        background: rgba(0, 255, 136, 0.15);
        color: {THEME_SUCCESS};
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid {THEME_SUCCESS};
        display: inline-block;
    }}
    .badge-pill {{
        background: rgba(0, 240, 255, 0.12);
        color: {THEME_PRIMARY};
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(0, 240, 255, 0.3);
    }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Sidebar
# ==============================================================================

st.sidebar.title("🔬 SynthQC")
st.sidebar.caption("AI-Assisted Visual Quality Inspection Prototype")
st.sidebar.markdown("---")

st.sidebar.subheader("System Configuration")
st.sidebar.write(f"**Target Category:** `{PRODUCT_CATEGORY}`")
st.sidebar.write(f"**Hardware Device:** `{DEVICE}`")
cgan_ready = Path("saved_models/patch_cgan_gen.pt").exists()
st.sidebar.markdown(f"**Patch cGAN:** <span class='badge-pill'>{'Ready' if cgan_ready else 'Unsaved'}</span>", unsafe_allow_html=True)
st.sidebar.write(f"**License:** Academic Non-Commercial (CC BY-NC-SA 4.0)")

counts = dm.get_subtype_counts()
st.sidebar.markdown("---")
st.sidebar.subheader("Data Partitioning")
st.sidebar.write(f"• **Train Good:** {counts.get('train_good', 0)}")
held_out = counts.get("held_out_test", {})
pool = counts.get("training_pool", {})
st.sidebar.write(f"• **Quarantined Test Defects:** {sum(v for k, v in held_out.items() if k != 'good')} (8 each)")
st.sidebar.write(f"• **Remaining Training Defects:** {sum(pool.values())}")
st.sidebar.write(f"• **Quarantined Test Good:** {held_out.get('good', 0)}")

st.sidebar.markdown("---")
st.sidebar.markdown("[🔗 GitHub Repository](https://github.com/hsinha2004/SynthQC)")

tabs = st.tabs([
    "📦 1. Dataset View",
    "🧪 2. Synthetic Defect Studio",
    "⚡ 3. Training View",
    "🔍 4. Inspection & QA",
    "📊 5. Quality Dashboard"
])


# ==============================================================================
# TAB 1: DATASET VIEW
# ==============================================================================
with tabs[0]:
    st.markdown('<div class="main-header">Dataset Explorer & Partitioning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect product images, ground truth defect masks, and verified data partitions.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{counts.get("train_good", 0)}</div><div class="stat-lbl">Train Good Images</div></div>', unsafe_allow_html=True)
    with col2:
        test_defect_sum = sum(v for k, v in held_out.items() if k != 'good')
        st.markdown(f'<div class="stat-card"><div class="stat-val">{test_defect_sum}</div><div class="stat-lbl">Quarantined Test Defects</div></div>', unsafe_allow_html=True)
    with col3:
        pool_defect_sum = sum(pool.values())
        st.markdown(f'<div class="stat-card"><div class="stat-val">{pool_defect_sum}</div><div class="stat-lbl">Remaining Training Pool</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{held_out.get("good", 0)}</div><div class="stat-lbl">Quarantined Test Good</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Subtype Breakdown")
    subtypes_df = pd.DataFrame([
        {"Subtype": "broken_large", "Quarantined Held-out Test": held_out.get("broken_large", 0), "Remaining Training Pool": pool.get("broken_large", 0), "Total Real": held_out.get("broken_large", 0) + pool.get("broken_large", 0)},
        {"Subtype": "broken_small", "Quarantined Held-out Test": held_out.get("broken_small", 0), "Remaining Training Pool": pool.get("broken_small", 0), "Total Real": held_out.get("broken_small", 0) + pool.get("broken_small", 0)},
        {"Subtype": "contamination", "Quarantined Held-out Test": held_out.get("contamination", 0), "Remaining Training Pool": pool.get("contamination", 0), "Total Real": held_out.get("contamination", 0) + pool.get("contamination", 0)},
        {"Subtype": "good", "Quarantined Held-out Test": held_out.get("good", 0), "Remaining Training Pool": counts.get("train_good", 0), "Total Real": held_out.get("good", 0) + counts.get("train_good", 0)},
    ])
    st.dataframe(subtypes_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Interactive Image & Ground Truth Mask Browser")

    browse_partition = st.radio(
        "Select Partition to Browse:",
        ["Remaining Training Pool (Allottable)", "Quarantined Held-Out Test Set", "Good Training Set"],
        horizontal=True
    )

    if browse_partition == "Good Training Set":
        good_imgs = dm.get_train_good_images()
        if good_imgs:
            idx = st.slider("Select Good Image Index:", 0, len(good_imgs) - 1, 0)
            img_p = good_imgs[idx]
            st.image(img_p, caption=f"Good Train Image: {Path(img_p).name}", width=400)
    else:
        items_to_browse = dm.split_registry.get("training_pool" if "Remaining" in browse_partition else "held_out_test", [])
        if items_to_browse:
            subtypes_avail = sorted(list(set(it["defect_type"] for it in items_to_browse)))
            chosen_sub = st.selectbox("Filter by Subtype:", subtypes_avail)
            filtered = [it for it in items_to_browse if it["defect_type"] == chosen_sub]

            if filtered:
                sel_idx = st.slider(f"Select Image ({chosen_sub}):", 0, len(filtered) - 1, 0)
                item = filtered[sel_idx]

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    st.write(f"**Original Product Image** (`{item.get('stem')}.png`)")
                    if os.path.exists(item["img_path"]):
                        st.image(item["img_path"], use_container_width=True)
                with b_col2:
                    st.write(f"**Ground Truth Defect Mask**")
                    if item.get("mask_path") and os.path.exists(item["mask_path"]):
                        st.image(item["mask_path"], use_container_width=True)
                    else:
                        st.info("No defect mask (sample is normal/good).")


# ==============================================================================
# TAB 2: SYNTHETIC DEFECT STUDIO
# ==============================================================================
with tabs[1]:
    st.markdown('<div class="main-header">Synthetic Defect Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive placement controls, Poisson blending, 64x64 patch cGAN, and 3-way visual comparisons.</div>', unsafe_allow_html=True)

    studio_mode = st.radio(
        "Studio Mode:",
        ["🎯 Interactive Placement Studio", "🎲 Automated Batch Generator", "⚔️ 3-Way Generator Showdown (Real vs Proc vs GAN)"],
        horizontal=True
    )

    good_imgs = dm.get_train_good_images()

    if studio_mode == "🎯 Interactive Placement Studio":
        s_col1, s_col2 = st.columns([1, 2])

        with s_col1:
            st.subheader("Studio Controls")
            studio_frac = st.select_slider(
                "Allotted Defect Source Pool:",
                options=[0.10, 0.25, 0.50, 1.0],
                value=0.25,
                format_func=lambda x: f"{int(x*100)}% ({max(1, int(round(39*x)))} real images)"
            )
            allotted_defects = dm.sample_training_defects(studio_frac, 42)
            st.caption(f"🔒 Zero-leakage: drawing from {len(allotted_defects)} allotted training defect images.")

            proc_gen = ProceduralGenerator(allotted_defects, good_imgs)

            chosen_blend = st.selectbox("Blending Method:", ["poisson_normal", "poisson_mixed", "alpha"])
            pos_x = st.slider("X Position (Center Coordinate):", 120, 380, 250, step=5)
            pos_y = st.slider("Y Position (Center Coordinate):", 120, 380, 250, step=5)
            scale_val = st.slider("Patch Scale:", 0.5, 1.6, 1.0, step=0.05)
            rot_val = st.slider("Rotation Angle (degrees):", -180, 180, 0, step=5)
            feather_val = st.slider("Edge Feathering Radius:", 3, 25, 11, step=2)

            patch_idx = st.number_input("Select Patch Index:", min_value=0, max_value=max(0, len(proc_gen.patches) - 1), value=0)
            good_idx = st.number_input("Select Base Good Image Index:", min_value=0, max_value=max(0, len(good_imgs) - 1), value=0)

        with s_col2:
            st.subheader("Live Studio Synthesis Preview")
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
                    st.image(meta["good_rgb"], caption="1. Clean Base Product", use_container_width=True)
                with p2:
                    st.image(meta["patch_img"], caption=f"2. Defect Patch ({dtype})", use_container_width=True)
                with p3:
                    st.image(synth_rgb, caption="3. Blended Defective Result", use_container_width=True)
                with p4:
                    st.image(synth_mask, caption="4. Pixel Defect Mask", use_container_width=True)

                st.markdown("---")
                if st.button("📥 Add Sample Directly to Synthetic Training Pool", type="primary"):
                    p_img, p_mask = inject_interactive_sample(synth_rgb, synth_mask, prefix="studio_interactive")
                    st.success(f"Sample added to pool! (`{Path(p_img).name}`)")

    elif studio_mode == "🎲 Automated Batch Generator":
        b_col1, b_col2 = st.columns([1, 2])
        with b_col1:
            st.subheader("Batch Generation Parameters")
            b_frac = st.select_slider("Allotted Pool Fraction:", options=[0.10, 0.25, 0.50, 1.0], value=0.50)
            b_count = st.slider("Number of Synthetic Samples to Generate:", 5, 50, 20)
            b_blend = st.selectbox("Batch Blending Mode:", ["poisson_normal", "poisson_mixed", "alpha"])
            b_seed = st.number_input("Random Seed:", value=42, step=1)
            b_gen_btn = st.button("🚀 Generate Synthetic Batch", type="primary")

        with b_col2:
            st.subheader("Batch Generation Output")
            if b_gen_btn:
                allotted_defects = dm.sample_training_defects(b_frac, b_seed)
                proc_gen = ProceduralGenerator(allotted_defects, good_imgs)
                out_dir = dm.synthetic_dir / f"batch_gen_{b_seed}"

                with st.spinner(f"Generating {b_count} synthetic defect samples..."):
                    paths = proc_gen.generate_batch(b_count, out_dir, prefix="batch_synth", rng_seed=b_seed)

                st.success(f"Generated {len(paths)} synthetic defective images and masks in `{out_dir.name}`!")

                # Display preview grid
                preview_paths = paths[:6]
                g_cols = st.columns(min(len(preview_paths), 3))
                for i, p in enumerate(preview_paths):
                    col = g_cols[i % len(g_cols)]
                    col.image(p, caption=Path(p).name, use_container_width=True)

    elif studio_mode == "⚔️ 3-Way Generator Showdown (Real vs Proc vs GAN)":
        st.subheader("Generator Comparison: Real vs. Procedural Poisson vs. 64x64 Patch cGAN")
        st.caption("Inspect how the different defect synthesis methods compare against an authentic real defective product.")

        allotted_defects = dm.sample_training_defects(1.0, 42)
        proc_gen = ProceduralGenerator(allotted_defects, good_imgs)
        gan_trainer = PatchGANTrainer(allotted_defects, good_imgs)
        cgan_loaded = gan_trainer.load_pretrained("saved_models/patch_cgan_gen.pt")

        c1, c2, c3 = st.columns(3)
        # 1. Real Defect
        with c1:
            st.markdown("#### 1. Real Defect (Ground Truth)")
            if allotted_defects:
                sample_real = allotted_defects[0]
                st.image(sample_real["img_path"], caption=f"Real Defect: {sample_real['defect_type']}", use_container_width=True)
                if sample_real.get("mask_path") and os.path.exists(sample_real["mask_path"]):
                    st.image(sample_real["mask_path"], caption="Real Ground-Truth Mask", use_container_width=True)

        # 2. Procedural Poisson Defect
        with c2:
            st.markdown("#### 2. Tier A: Procedural Poisson Blend")
            s_proc, s_mask_proc, p_type, _ = proc_gen.generate_single(blend_mode="poisson_normal")
            if s_proc is not None:
                st.image(s_proc, caption=f"Procedural Defect: {p_type}", use_container_width=True)
                st.image(s_mask_proc, caption="Procedural Synthetic Mask", use_container_width=True)

        # 3. Patch cGAN Defect
        with c3:
            st.markdown("#### 3. Tier B: 64x64 Patch cGAN (Experimental)")
            if cgan_loaded:
                s_gan, s_mask_gan, gan_meta = gan_trainer.generate_full_synthetic_image(good_imgs[0])
                if s_gan is not None:
                    st.image(s_gan, caption="Patch cGAN Composite", use_container_width=True)
                    st.image(s_mask_gan, caption="cGAN Defect Mask", use_container_width=True)
                else:
                    st.info("Could not composite cGAN image.")
            else:
                st.warning("Pre-trained cGAN model not found. Train in Tab 3 or check saved_models/patch_cgan_gen.pt.")


# ==============================================================================
# TAB 3: TRAINING VIEW
# ==============================================================================
with tabs[2]:
    st.markdown('<div class="main-header">Multi-Arm Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Train ResNet18 across experimental arms with strict split quarantine and defect scarcity simulation.</div>', unsafe_allow_html=True)

    t_col1, t_col2 = st.columns([1, 2])

    with t_col1:
        st.subheader("Experiment Parameters")

        train_frac = st.select_slider(
            "Real Defect Scarcity Level (% of Remaining Pool):",
            options=[0.10, 0.25, 0.50, 1.0],
            value=0.25,
            format_func=lambda x: f"{int(x*100)}% ({max(1, int(round(39*x)))} real images)"
        )

        num_splits = st.slider("Number of Random Splits (for mean ± std):", min_value=2, max_value=5, value=3)
        epochs_per_run = st.slider("Epochs per Split:", min_value=3, max_value=10, value=5)

        st.markdown("---")
        st.write("**Select Arms to Train:**")
        train_arm_a = st.checkbox("Arm A (Real Defect Images Only)", value=True)
        train_arm_b = st.checkbox("Arm B (Real + Procedural Synthetic)", value=True)
        train_arm_c = st.checkbox("Arm C (Real + Patch-GAN Synthetic)", value=False)

        st.markdown("---")
        sanity_btn = st.button("🧪 Run Arm A Sanity Check First")
        train_btn = st.button("🚀 Train Selected Arms", type="primary")

    with t_col2:
        st.subheader("Training Progress & Live Metrics")

        if sanity_btn:
            with st.spinner("Executing Sanity Check on Arm A (100% pool across 3 splits)..."):
                sanity_res = run_sanity_check(runner)
                st.session_state.sanity_check_status = sanity_res

        if st.session_state.sanity_check_status:
            sc = st.session_state.sanity_check_status
            if sc["passed"]:
                st.success(f"✅ **Sanity Check Passed:** Arm A achieves non-trivial classification (Balanced Accuracy: **{sc['balanced_accuracy_mean']*100:.1f}%**, F1: **{sc['f1_mean']*100:.1f}%**). Cross-arm comparison is scientifically valid.")
            else:
                st.error(f"❌ **Sanity Check Failed:** Arm A hovered near chance ({sc['balanced_accuracy_mean']*100:.1f}%). Adjust hyper-parameters before running full comparisons.")

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
                st.warning("Please select at least one arm to train.")
            else:
                for a_idx, arm_name in enumerate(arms_to_run):
                    status_text.write(f"Training {arm_name} across {num_splits} splits...")

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
                status_text.success("🎉 All selected training runs completed successfully!")

        if st.session_state.benchmark_results:
            st.markdown("#### Completed Benchmark Runs")
            display_rows = []
            for r in st.session_state.benchmark_results[-8:]:
                display_rows.append({
                    "Arm": r["arm_name"],
                    "Real Data %": f"{int(r['fraction']*100)}%",
                    "Real Defects": r["allotted_defect_count"],
                    "Precision": f"{r['precision_mean']*100:.1f} ± {r['precision_std']*100:.1f}%",
                    "Recall": f"{r['recall_mean']*100:.1f} ± {r['recall_std']*100:.1f}%",
                    "F1 Score": f"{r['f1_mean']*100:.1f} ± {r['f1_std']*100:.1f}%",
                    "Balanced Acc": f"{r['balanced_accuracy_mean']*100:.1f} ± {r['balanced_accuracy_std']*100:.1f}%",
                    "ROC-AUC": f"{r['roc_auc_mean']:.3f} ± {r['roc_auc_std']:.3f}",
                })
            st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)


# ==============================================================================
# TAB 4: INSPECTION & QA STATION
# ==============================================================================
with tabs[3]:
    st.markdown('<div class="main-header">Live Inspection & Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Run product images through trained models, inspect Grad-CAM attention localization with IoU scores, and record human QA audits.</div>', unsafe_allow_html=True)

    inspect_mode = st.radio(
        "Inspection Scope:",
        ["🔍 Single Image Detailed Inspection", "📑 Batch Test Inspection Grid"],
        horizontal=True
    )

    avail_models = list(st.session_state.trained_models.keys())
    if not avail_models:
        st.info("⚠️ No custom trained models yet in session. Defaulting to pre-trained ResNet18 baseline.")
        selected_model_name = "Baseline Pretrained ResNet18"
        active_model = create_model()
    else:
        selected_model_name = st.selectbox("Active Inspection Model:", avail_models)
        active_model = st.session_state.trained_models[selected_model_name]

    test_items = dm.get_fixed_test_items()

    if inspect_mode == "🔍 Single Image Detailed Inspection":
        ins_col1, ins_col2 = st.columns([1, 2])

        with ins_col1:
            st.subheader("Image Selection")
            img_source = st.radio("Source:", ["Select from Quarantined Held-Out Test Set", "Upload Custom Image"])

            selected_img_path = None
            ground_truth_label = None
            mask_path_for_iou = None

            if img_source == "Select from Quarantined Held-Out Test Set":
                test_subtypes = sorted(list(set(it[2] for it in test_items)))
                sel_st = st.selectbox("Subtype Filter:", test_subtypes)
                matching = [it for it in test_items if it[2] == sel_st]
                sel_idx = st.slider("Sample Index:", 0, len(matching) - 1, 0)
                selected_img_path = matching[sel_idx][0]
                ground_truth_label = "Defective" if matching[sel_idx][1] == 1 else "Normal (Good)"

                # Find mask path if defective
                if matching[sel_idx][1] == 1:
                    stem = Path(selected_img_path).stem
                    possible_mask = dm.gt_dir / sel_st / f"{stem}_mask.png"
                    if possible_mask.exists():
                        mask_path_for_iou = str(possible_mask)
            else:
                uploaded = st.file_uploader("Upload Product Image:", type=["png", "jpg", "jpeg"])
                if uploaded:
                    save_path = dm.category_dir / f"custom_upload_{uploaded.name}"
                    with open(save_path, "wb") as f:
                        f.write(uploaded.read())
                    selected_img_path = str(save_path)
                    ground_truth_label = "Unknown (User Upload)"

            run_inspect_btn = st.button("🔍 Run Visual Inspection", type="primary")

        with ins_col2:
            st.subheader("Inference Result & Grad-CAM Localization")

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
                        st.markdown(f'<div class="flag-defect">🚨 DEFECT FLAGGED (Confidence: {defect_prob*100:.1f}%)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="flag-pass">✅ PASS / NORMAL (Confidence: {(1.0-defect_prob)*100:.1f}%)</div>', unsafe_allow_html=True)

                with r_col2:
                    st.write(f"**Ground Truth:** `{ground_truth_label}`")
                    if iou_res.get("has_gt") and iou_res.get("iou") is not None:
                        st.markdown(f"**Grad-CAM Localization IoU:** <span class='badge-pill'>{iou_res['iou']*100:.1f}%</span> (Dice: {iou_res['dice']*100:.1f}%)", unsafe_allow_html=True)

                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.image(orig_rgb, caption="Original Product Image", use_container_width=True)
                with v_col2:
                    st.image(overlay, caption="Grad-CAM Attention Overlay", use_container_width=True)

                st.markdown("---")
                st.subheader("🧑‍🏭 Human-in-the-Loop QA Station")
                qa_col1, qa_col2, qa_col3 = st.columns(3)
                with qa_col1:
                    if st.button("👍 Approve AI Decision", use_container_width=True):
                        st.session_state.inspection_log.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image": Path(selected_img_path).name,
                            "model": selected_model_name,
                            "defect_prob": f"{defect_prob*100:.1f}%",
                            "ai_decision": "Defect" if defect_prob >= st.session_state.operating_threshold else "Good",
                            "human_action": "Approved",
                            "ground_truth": ground_truth_label,
                            "iou": f"{iou_res['iou']*100:.1f}%" if iou_res.get("has_gt") else "N/A"
                        })
                        st.success("Decision approved and recorded.")
                with qa_col2:
                    if st.button("🔄 Override Decision", use_container_width=True):
                        st.session_state.inspection_log.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image": Path(selected_img_path).name,
                            "model": selected_model_name,
                            "defect_prob": f"{defect_prob*100:.1f}%",
                            "ai_decision": "Defect" if defect_prob >= st.session_state.operating_threshold else "Good",
                            "human_action": "Overridden",
                            "ground_truth": ground_truth_label,
                            "iou": f"{iou_res['iou']*100:.1f}%" if iou_res.get("has_gt") else "N/A"
                        })
                        st.warning("Decision overridden and recorded.")
                with qa_col3:
                    if st.button("⚠️ Flag for Physical QA", use_container_width=True):
                        st.session_state.inspection_log.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image": Path(selected_img_path).name,
                            "model": selected_model_name,
                            "defect_prob": f"{defect_prob*100:.1f}%",
                            "ai_decision": "Defect" if defect_prob >= st.session_state.operating_threshold else "Good",
                            "human_action": "Flagged For Physical QA",
                            "ground_truth": ground_truth_label,
                            "iou": f"{iou_res['iou']*100:.1f}%" if iou_res.get("has_gt") else "N/A"
                        })
                        st.info("Flagged for physical QA review.")

    elif inspect_mode == "📑 Batch Test Inspection Grid":
        st.subheader("Batch Testing on Quarantined Held-Out Test Set")
        st.caption("Inspect how the model performs across a batch of test images simultaneously.")

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

        batch_limit = st.slider("Number of images to inspect in batch:", 4, len(batch_items), 12, step=4)
        run_batch_btn = st.button("🚀 Run Batch Inspection", type="primary")

        if run_batch_btn:
            with st.spinner("Running batch inference & generating Grad-CAM heatmaps..."):
                batch_results = batch_inspect_images(active_model, batch_items[:batch_limit], threshold=st.session_state.operating_threshold)

            # Summary metrics
            total = len(batch_results)
            flagged = sum(1 for r in batch_results if r["is_flagged"])
            ious = [r["iou"] for r in batch_results if r.get("iou") is not None]
            mean_iou = float(np.mean(ious)) if ious else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Tested", total)
            m2.metric("Flagged as Defective", flagged)
            m3.metric("Cleared as Normal", total - flagged)
            m4.metric("Mean Localization IoU", f"{mean_iou*100:.1f}%")

            st.markdown("---")
            # Render grid
            cols_per_row = 3
            for r_i in range(0, len(batch_results), cols_per_row):
                row_items = batch_results[r_i:r_i + cols_per_row]
                row_cols = st.columns(cols_per_row)
                for c_i, r in enumerate(row_items):
                    with row_cols[c_i]:
                        status_str = "🚨 DEFECT" if r["is_flagged"] else "✅ PASS"
                        iou_disp = f"{r['iou']*100:.1f}%" if r.get("iou") is not None else "N/A"
                        st.image(r["overlay_rgb"], caption=f"{r['stem']} | {status_str} ({r['defect_prob']*100:.1f}%)", use_container_width=True)
                        st.caption(f"Ground Truth: `{r['defect_type']}` | IoU: `{iou_disp}`")


# ==============================================================================
# TAB 5: DASHBOARD & OPERATING THRESHOLD TUNER
# ==============================================================================
with tabs[4]:
    st.markdown('<div class="main-header">Quality Dashboard & Decision Threshold Tuner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Objective empirical evaluation of data efficiency across experimental arms and interactive inspection threshold tuning.</div>', unsafe_allow_html=True)

    # Threshold Tuner Section
    st.subheader("🎚️ Interactive Inspection Sensitivity (Decision Threshold)")
    st.caption("Adjust the operating threshold to explore the trade-off between False Alarms and Missed Defects on the held-out test set.")

    t_thresh = st.slider(
        "Classification Threshold (P(Defect) >= Threshold flags product):",
        min_value=0.05, max_value=0.95, value=float(st.session_state.operating_threshold), step=0.05
    )
    st.session_state.operating_threshold = t_thresh

    # Compute threshold metrics
    if avail_models:
        sel_m = st.session_state.trained_models.get(list(st.session_state.trained_models.keys())[0], None)
        if sel_m is not None:
            t_eval = evaluate_at_threshold(sel_m, test_items, threshold=t_thresh)

            tm1, tm2, tm3, tm4 = st.columns(4)
            tm1.metric("Precision", f"{t_eval['precision']*100:.1f}%", help="TP / (TP + FP)")
            tm2.metric("Recall (Sensitivity)", f"{t_eval['recall']*100:.1f}%", help="TP / (TP + FN)")
            tm3.metric("Specificity", f"{t_eval['specificity']*100:.1f}%", help="TN / (TN + FP)")
            tm4.metric("False Alarm Rate", f"{t_eval['false_alarm_rate']*100:.1f}%", help="FP / (TN + FP)")

            # Confusion Matrix Cards
            c_cm1, c_cm2 = st.columns([1, 1])
            with c_cm1:
                st.markdown(f"""
                | Confusion Matrix | Actual Normal (0) | Actual Defective (1) |
                | :--- | :---: | :---: |
                | **AI Predicted Normal** | **TN: {t_eval['tn']}** | FN (Missed): {t_eval['fn']} |
                | **AI Predicted Defect** | FP (False Alarm): {t_eval['fp']} | **TP: {t_eval['tp']}** |
                """)
            with c_cm2:
                # Dynamic ROC curve
                fpr = t_eval['false_alarm_rate']
                tpr = t_eval['recall']
                fig_roc, ax_roc = plt.subplots(figsize=(6, 3), facecolor='#0B0F19')
                ax_roc.set_facecolor('#161E31')
                ax_roc.plot([0, 1], [0, 1], '--', color='gray')
                ax_roc.plot([0, fpr, 1], [0, tpr, 1], color='#00F0FF', lw=2, label=f"AUC = {t_eval['roc_auc']:.3f}")
                ax_roc.plot([fpr], [tpr], 'o', color='#FF3366', markersize=9, label=f"Threshold {t_thresh:.2f}")
                ax_roc.set_xlabel('False Positive Rate', color='white')
                ax_roc.set_ylabel('True Positive Rate', color='white')
                ax_roc.set_title('Operating Point on ROC', color='white', fontsize=10)
                ax_roc.tick_params(colors='white')
                ax_roc.legend(facecolor='#0B0F19', labelcolor='white')
                st.pyplot(fig_roc)
                plt.close()

    st.markdown("---")

    # Benchmark Table
    if st.session_state.benchmark_results:
        st.subheader("Quantitative Cross-Arm Evaluation (Held-out Test Set)")
        table_rows = []
        for r in st.session_state.benchmark_results:
            table_rows.append({
                "Arm": r["arm_name"],
                "Real Defect %": int(r["fraction"] * 100),
                "Real Count": r["allotted_defect_count"],
                "Precision (Mean ± Std)": f"{r['precision_mean']*100:.1f} ± {r['precision_std']*100:.1f}%",
                "Recall (Mean ± Std)": f"{r['recall_mean']*100:.1f} ± {r['recall_std']*100:.1f}%",
                "F1 Score (Mean ± Std)": f"{r['f1_mean']*100:.1f} ± {r['f1_std']*100:.1f}%",
                "Balanced Acc (Mean ± Std)": f"{r['balanced_accuracy_mean']*100:.1f} ± {r['balanced_accuracy_std']*100:.1f}%",
                "ROC-AUC (Mean ± Std)": f"{r['roc_auc_mean']:.3f} ± {r['roc_auc_std']:.3f}",
            })

        df_bench = pd.DataFrame(table_rows)
        st.dataframe(df_bench, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Data-Efficiency Curves: Performance vs. % Real Defect Data")
        st.caption("Answers the empirical question: How does defect detection performance scale as real defective examples become scarce?")

        plot_data = []
        for r in st.session_state.benchmark_results:
            plot_data.append({
                "Real Defect Fraction (%)": int(r["fraction"] * 100),
                "F1 Score": r["f1_mean"] * 100.0,
                "Balanced Accuracy": r["balanced_accuracy_mean"] * 100.0,
                "Recall": r["recall_mean"] * 100.0,
                "Arm": r["arm_name"]
            })

        df_plot = pd.DataFrame(plot_data)
        metric_to_plot = st.selectbox("Select Metric to Plot:", ["F1 Score", "Balanced Accuracy", "Recall"])

        pivot_df = df_plot.pivot_table(
            index="Real Defect Fraction (%)",
            columns="Arm",
            values=metric_to_plot,
            aggfunc="mean"
        ).sort_index()

        st.line_chart(pivot_df)

    # Human Review Log & CSV Export
    st.markdown("---")
    st.subheader("Human QA Review Audit Log")
    if st.session_state.inspection_log:
        df_log = pd.DataFrame(st.session_state.inspection_log)
        st.dataframe(df_log, hide_index=True, use_container_width=True)

        csv_buffer = io.StringIO()
        df_log.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export QA Audit Log (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"synthqc_qa_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.caption("No manual inspections recorded in this session yet. Use Tab 4 (Inspection View) to review and log product decisions.")

"""
SynthQC - AI-Assisted Visual Quality Inspection Web Application
Single multi-tab Streamlit application:
1. 📦 Dataset View
2. 🧪 Synthetic Defect Lab
3. ⚡ Training View (Multi-Arm with Scarcity Simulation & Zero Leakage)
4. 🔍 Inspection View (Inference, Grad-CAM Explainability & Human QA Station)
5. 📊 Dashboard View (Objective Metrics, Data-Efficiency Curves & QA Audit)
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch

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
from synthetic_generator import ProceduralGenerator, PatchGANTrainer, extract_defect_patches
from models import create_model, GradCAM, ExperimentRunner, run_sanity_check, train_single_run

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
            m.load_state_dict(torch.load(model_path, map_location=DEVICE))
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

dm = st.session_state.data_manager
runner = st.session_state.experiment_runner

# ==============================================================================
# CSS & Styling (Applied subtly for clean industrial UI)
# ==============================================================================

st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }}
    .sub-header {{
        color: #8E9BB0;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }}
    .stat-card {{
        background: {THEME_CARD};
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }}
    .stat-val {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {THEME_PRIMARY};
    }}
    .stat-lbl {{
        font-size: 0.85rem;
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
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Sidebar & Header
# ==============================================================================

st.sidebar.title("🔬 SynthQC")
st.sidebar.caption("AI-Assisted Visual Quality Inspection Prototype")
st.sidebar.markdown("---")

st.sidebar.subheader("System Configuration")
st.sidebar.write(f"**Target Category:** `{PRODUCT_CATEGORY}` (MVTec AD)")
st.sidebar.write(f"**Hardware Device:** `{DEVICE}`")
st.sidebar.write(f"**License:** Academic Non-Commercial (CC BY-NC-SA 4.0)")

# Quick dataset stats in sidebar
counts = dm.get_subtype_counts()
st.sidebar.markdown("---")
st.sidebar.subheader("Data Partitioning")
st.sidebar.write(f"• **Train Good:** {counts.get('train_good', 0)}")
held_out = counts.get("held_out_test", {})
pool = counts.get("training_pool", {})
st.sidebar.write(f"• **Held-out Test Defective:** {sum(v for k, v in held_out.items() if k != 'good')} (8 each)")
st.sidebar.write(f"• **Remaining Training Defects:** {sum(pool.values())}")
st.sidebar.write(f"• **Held-out Test Good:** {held_out.get('good', 0)}")

tabs = st.tabs([
    "📦 1. Dataset View",
    "🧪 2. Synthetic Defect Lab",
    "⚡ 3. Training View",
    "🔍 4. Inspection View",
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
# TAB 2: SYNTHETIC DEFECT LAB
# ==============================================================================
with tabs[1]:
    st.markdown('<div class="main-header">Synthetic Defect Generation Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Generate procedural (cut-paste/Poisson) or experimental patch-GAN defects strictly from allotted training samples (zero test leakage).</div>', unsafe_allow_html=True)

    gen_col1, gen_col2 = st.columns([1, 2])

    with gen_col1:
        st.subheader("Generation Settings")
        tier_mode = st.radio(
            "Generator Tier:",
            ["Tier A: Procedural Poisson Blend (Baseline)", "Tier B: 64x64 Patch cGAN (Experimental)"]
        )

        allotted_frac = st.select_slider(
            "Allotted Real Defect Fraction:",
            options=[0.10, 0.25, 0.50, 1.0],
            value=0.25,
            format_func=lambda x: f"{int(x*100)}% ({max(1, int(round(39*x)))} real images)"
        )
        split_seed = st.number_input("Split Seed (for allotted subset):", value=42, step=1)

        allotted_for_lab = dm.sample_training_defects(allotted_frac, int(split_seed))
        st.info(f"🔒 **Zero Leakage Active**: Only {len(allotted_for_lab)} real defect images from the training pool are accessible to this generator. Held-out test set is strictly quarantined.")

        if "Procedural" in tier_mode:
            blend_mode = st.selectbox(
                "Blending Method:",
                ["poisson_normal", "poisson_mixed", "alpha"]
            )
            scale_min, scale_max = st.slider("Scale Range:", 0.5, 1.5, (0.8, 1.2))
            rot_limit = st.slider("Max Rotation Angle (degrees):", 0, 45, 25)
            opacity = st.slider("Blend Opacity (Alpha fallback):", 0.5, 1.0, 1.0)
        else:
            st.warning("⚠️ **Tier B is an experimental stretch goal.** Trains a 64x64 U-Net + PatchGAN on cropped defect patches. Generated quality will be evaluated honestly.")
            gan_epochs = st.slider("GAN Training Epochs (CPU):", 5, 20, 8)
            train_gan_btn = st.button("Train / Verify 64x64 Patch GAN")

    with gen_col2:
        st.subheader("Generation Preview & Source Traceability")
        good_imgs = dm.get_train_good_images()

        if "Procedural" in tier_mode:
            proc_gen = ProceduralGenerator(allotted_for_lab, good_imgs)

            preview_btn = st.button("🎲 Generate Single Procedural Preview")
            if preview_btn or "last_proc_preview" not in st.session_state:
                synth_rgb, synth_mask, dtype = proc_gen.generate_single(
                    blend_mode=blend_mode,
                    scale_range=(scale_min, scale_max),
                    rotation_max=rot_limit,
                    opacity=opacity
                )
                st.session_state.last_proc_preview = (synth_rgb, synth_mask, dtype)

            if "last_proc_preview" in st.session_state:
                s_rgb, s_mask, s_dtype = st.session_state.last_proc_preview
                if s_rgb is not None:
                    p_c1, p_c2 = st.columns(2)
                    with p_c1:
                        st.image(s_rgb, caption=f"Synthetic Defect Image (Subtype: {s_dtype})", use_container_width=True)
                    with p_c2:
                        st.image(s_mask, caption="Synthetic Ground Truth Mask", use_container_width=True)
                else:
                    st.error("Could not generate synthetic image. Check source images.")

        else:
            # Tier B Patch cGAN
            if "gan_trainer_state" not in st.session_state:
                st.session_state.gan_trainer_state = None

            if train_gan_btn:
                with st.spinner("Training 64x64 Patch cGAN on allotted defect patches..."):
                    trainer = PatchGANTrainer(allotted_for_lab, good_imgs)
                    res = trainer.train_quick(epochs=gan_epochs)
                    st.session_state.gan_trainer_state = (trainer, res)

            if st.session_state.gan_trainer_state is not None:
                trainer, res = st.session_state.gan_trainer_state
                st.write(f"**GAN Status:** `{res['status']}`")
                st.write(f"**Diagnostic Check:** {res['diagnostic_message']}")

                if res["is_degenerate"]:
                    st.error("🚨 **Honest Failure Report:** The GAN generated degenerate / collapsed patches. In accordance with project requirements, these will not be silently included as valid synthetic training samples.")
                else:
                    st.success("✅ **Quality Check Passed:** Patch spatial variance is non-degenerate.")

                # Show sample generated patch
                sample_p_btn = st.button("Generate Test Patch Sample")
                if sample_p_btn:
                    # Pick sample
                    if trainer.patches:
                        sample_p = trainer.patches[0]
                        dummy_good = np.full((64, 64, 3), 180, dtype=np.uint8)
                        gen_patch = trainer.sample_patch(dummy_good, sample_p["patch_mask"])

                        c_g1, c_g2, c_g3 = st.columns(3)
                        with c_g1:
                            st.image(sample_p["patch_img"], caption="Real Source Patch", width=128)
                        with c_g2:
                            st.image(sample_p["patch_mask"], caption="Defect Mask", width=128)
                        with c_g3:
                            st.image(gen_patch, caption="cGAN Generated Patch", width=128)


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
        # Sanity Check Button
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

                    # Store in session state
                    st.session_state.benchmark_results.append(res)
                    st.session_state.trained_models[f"{arm_name} [{int(train_frac*100)}%]"] = res["best_model"]

                progress_bar.progress(1.0)
                status_text.success("🎉 All selected training runs completed successfully!")

        # Display currently trained models and recent runs
        if st.session_state.benchmark_results:
            st.markdown("#### Completed Benchmark Runs")
            display_rows = []
            for r in st.session_state.benchmark_results[-6:]:
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
# TAB 4: INSPECTION VIEW & GRAD-CAM
# ==============================================================================
with tabs[3]:
    st.markdown('<div class="main-header">Live Inspection & Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Run product images through trained models, inspect Grad-CAM attention localization, and submit human QA reviews.</div>', unsafe_allow_html=True)

    ins_col1, ins_col2 = st.columns([1, 2])

    with ins_col1:
        st.subheader("Inspection Controls")

        avail_models = list(st.session_state.trained_models.keys())
        if not avail_models:
            st.info("⚠️ No custom trained models yet in session. Defaulting to pre-trained ResNet18 baseline.")
            selected_model_name = "Baseline Pretrained ResNet18"
            active_model = create_model()
        else:
            selected_model_name = st.selectbox("Select Inspection Model:", avail_models)
            active_model = st.session_state.trained_models[selected_model_name]

        img_source = st.radio("Image Source:", ["Select from Held-Out Test Set", "Upload Custom Image"])

        test_items = dm.get_fixed_test_items()
        selected_img_path = None
        ground_truth_label = None

        if img_source == "Select from Held-Out Test Set":
            test_subtypes = sorted(list(set(it[2] for it in test_items)))
            sel_st = st.selectbox("Subtype Filter:", test_subtypes)
            matching = [it for it in test_items if it[2] == sel_st]
            sel_idx = st.slider("Select Sample Index:", 0, len(matching) - 1, 0)
            selected_img_path = matching[sel_idx][0]
            ground_truth_label = "Defective" if matching[sel_idx][1] == 1 else "Normal (Good)"
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
        st.subheader("Inference Result & Localization")

        if selected_img_path and os.path.exists(selected_img_path):
            img_pil = Image.open(selected_img_path).convert("RGB")
            orig_rgb = np.array(img_pil)

            # Perform Inference
            t_img = get_transforms(is_train=False)(img_pil).unsqueeze(0).to(DEVICE)
            active_model.eval()

            with torch.no_grad():
                output = active_model(t_img)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                pred_class = int(np.argmax(probs))
                defect_prob = float(probs[1])

            # Grad-CAM
            grad_cam = GradCAM(active_model)
            cam = grad_cam.generate_heatmap(t_img, class_idx=1)
            overlay = grad_cam.overlay_heatmap(orig_rgb, cam, alpha=0.55)

            # Display Status Badge
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                if pred_class == 1 or defect_prob >= 0.5:
                    st.markdown(f'<div class="flag-defect">🚨 DEFECT FLAGGED (Defect Prob: {defect_prob*100:.1f}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="flag-pass">✅ PASS / NORMAL (Good Prob: {(1.0-defect_prob)*100:.1f}%)</div>', unsafe_allow_html=True)

            with r_col2:
                st.write(f"**Ground Truth:** `{ground_truth_label}`")
                st.write(f"**Model Used:** `{selected_model_name}`")

            # Image & Heatmap Views
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.image(orig_rgb, caption="Original Product Image", use_container_width=True)
            with v_col2:
                st.image(overlay, caption="Grad-CAM Attention Overlay (Defect Focus)", use_container_width=True)

            st.markdown("---")
            st.subheader("🧑‍🏭 Human-in-the-Loop QA Station")
            st.caption("Inspect borderline confidence and record human review audits.")

            qa_col1, qa_col2, qa_col3 = st.columns(3)
            with qa_col1:
                if st.button("👍 Approve AI Decision", use_container_width=True):
                    st.session_state.inspection_log.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image": Path(selected_img_path).name,
                        "model": selected_model_name,
                        "defect_prob": f"{defect_prob*100:.1f}%",
                        "ai_decision": "Defect" if defect_prob >= 0.5 else "Good",
                        "human_action": "Approved",
                        "ground_truth": ground_truth_label
                    })
                    st.success("Decision approved and logged.")
            with qa_col2:
                if st.button("🔄 Override Decision", use_container_width=True):
                    st.session_state.inspection_log.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image": Path(selected_img_path).name,
                        "model": selected_model_name,
                        "defect_prob": f"{defect_prob*100:.1f}%",
                        "ai_decision": "Defect" if defect_prob >= 0.5 else "Good",
                        "human_action": "Overridden",
                        "ground_truth": ground_truth_label
                    })
                    st.warning("Decision overridden and logged.")
            with qa_col3:
                if st.button("⚠️ Needs Closer Inspection", use_container_width=True):
                    st.session_state.inspection_log.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image": Path(selected_img_path).name,
                        "model": selected_model_name,
                        "defect_prob": f"{defect_prob*100:.1f}%",
                        "ai_decision": "Defect" if defect_prob >= 0.5 else "Good",
                        "human_action": "Flagged For Physical QA",
                        "ground_truth": ground_truth_label
                    })
                    st.info("Flagged for physical QA review.")


# ==============================================================================
# TAB 5: DASHBOARD VIEW
# ==============================================================================
with tabs[4]:
    st.markdown('<div class="main-header">Data Efficiency & Quality Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Objective empirical evaluation of data efficiency across experimental arms and human QA audit logs.</div>', unsafe_allow_html=True)

    # If no benchmark runs yet, provide quick run button or seed with default runs
    if not st.session_state.benchmark_results:
        st.info("No multi-split benchmark results in session yet. Click below to run a standard baseline comparison across 10%, 25%, 50%, and 100% data scarcity levels.")
        quick_run_btn = st.button("⚡ Run Full Data-Efficiency Comparison (Arms A vs B)")
        if quick_run_btn:
            with st.spinner("Evaluating Arms A & B across all scarcity levels (10%, 25%, 50%, 100%)..."):
                for frac in [0.10, 0.25, 0.50, 1.0]:
                    res_a = runner.run_arm("Arm A (Real Only)", fraction=frac, num_splits=3, epochs_per_run=5)
                    res_b = runner.run_arm("Arm B (Real + Procedural)", fraction=frac, num_splits=3, epochs_per_run=5)
                    st.session_state.benchmark_results.extend([res_a, res_b])
                    st.session_state.trained_models[f"Arm A [{int(frac*100)}%]"] = res_a["best_model"]
                    st.session_state.trained_models[f"Arm B [{int(frac*100)}%]"] = res_b["best_model"]
            st.rerun()

    if st.session_state.benchmark_results:
        # Benchmark Table
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

        # Prepare chart data
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

        # Plot with Streamlit line chart / altair
        metric_to_plot = st.selectbox("Select Metric to Plot:", ["F1 Score", "Balanced Accuracy", "Recall"])

        pivot_df = df_plot.pivot_table(
            index="Real Defect Fraction (%)",
            columns="Arm",
            values=metric_to_plot,
            aggfunc="mean"
        ).sort_index()

        st.line_chart(pivot_df)

        st.markdown("---")
        st.subheader("Objective Empirical Findings")
        st.info("""
        **Methodological Notes:**
        - All evaluation metrics are calculated strictly on the quarantined, held-out test set (44 images: 20 good, 24 defective stratified across subtypes).
        - No test images or unallotted training samples were seen by synthetic generators.
        - Results represent mean ± standard deviation across independent random splits.
        - Examine whether procedural blending maintains recall at low fractions (10-25%) compared to real-only training.
        """)

    # Human Review Log
    st.markdown("---")
    st.subheader("Human QA Review Audit Log")
    if st.session_state.inspection_log:
        st.dataframe(pd.DataFrame(st.session_state.inspection_log), hide_index=True, use_container_width=True)
    else:
        st.caption("No manual inspections recorded in this session yet. Use Tab 4 (Inspection View) to review and log product decisions.")

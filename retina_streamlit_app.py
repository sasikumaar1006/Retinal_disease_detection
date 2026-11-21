from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

from gradcam import GradCAM, make_overlay

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

BASE_DIR = Path(__file__).resolve().parent

candidate_paths = [
    BASE_DIR / "oct_retinal_app" / "weights" / "resnet18_oct2017.pth",
    BASE_DIR / "weights" / "resnet18_oct2017.pth",
]

WEIGHTS_PATH = next((p for p in candidate_paths if p.exists()), candidate_paths[0])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(
    page_title="OCT Retinal Disease Classifier",
    page_icon="🩺",
    layout="wide",
)

# Elegant typography, centered title, saffron header, and off-white background
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;700&display=swap" rel="stylesheet">
    <style>
        html, body, .stApp, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #FAF8F5;
            color: #1E293B;
        }
        h1, h2, h3, h4, .saffron-title {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            letter-spacing: -0.015em;
        }
        .saffron-header {
            background: linear-gradient(90deg, #D95D00 0%, #F27D16 100%);
            color: #FFFFFF;
            padding: 1.4rem 1.8rem;
            border-radius: 12px;
            margin-bottom: 1.1rem;
            box-shadow: 0 3px 10px rgba(217, 93, 0, 0.16);
            text-align: center;
        }
        .saffron-title {
            color: #FFFFFF;
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            text-align: center;
        }
        .saffron-header div,
        .saffron-header p,
        .saffron-subtitle {
            display: block;
            width: 100%;
            max-width: 100%;
            margin-left: auto !important;
            margin-right: auto !important;
            text-align: center !important;
        }
        .saffron-subtitle {
            color: #FFF7ED;
            margin-top: 0.4rem !important;
            font-size: 0.98rem;
            font-weight: 400;
        }
        .dataset-card {
            background-color: #FFFFFF;
            border: 1px solid #E6E0D6;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .disclaimer-box {
            background-color: #FFF6EB;
            border-left: 5px solid #D95D00;
            color: #3B2314;
            padding: 0.85rem 1.15rem;
            border-radius: 8px;
            margin-top: 1.25rem;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .pred-orange-card {
            background: linear-gradient(135deg, #E86A0C 0%, #F5821F 100%);
            color: #FFFFFF;
            padding: 0.95rem 1.2rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(217, 93, 0, 0.2);
            margin-bottom: 1rem;
        }
        .pred-orange-label {
            font-size: 0.85rem;
            font-weight: 500;
            color: #FFF7ED;
            margin-bottom: 0.25rem;
        }
        .pred-orange-value {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="saffron-header">
        <h1 class="saffron-title">Retinal OCT Disease Classification &amp; Grad-CAM</h1>
        <p class="saffron-subtitle">
            Transfer Learning Evaluation &amp; Interactive Inference Prototype (CNV, DME, DRUSEN, NORMAL)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


def cam_to_colored_heatmap(cam: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    cam_resized = cv2.resize(cam, size, interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


@st.cache_resource
def load_resnet18_and_cam(weights_file: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    if weights_file.exists():
        state_dict = torch.load(weights_file, map_location=device)
        model.load_state_dict(state_dict)
    else:
        st.warning(f"Checkpoint not found at `{weights_file}`. Using uninitialized ResNet18 weights.")

    model.to(device)
    model.eval()
    cam_engine = GradCAM(model, target_layer=model.layer4[-1])
    return model, cam_engine, device


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

model, cam_engine, device = load_resnet18_and_cam(WEIGHTS_PATH)

tab1, tab2 = st.tabs([
    "1. Model Performance & Workflow",
    "2. Scan Upload & Prediction",
])

# ---------------------------------------------------------
# TAB 1: Dataset, Proportions, Model Comparisons & Workflow
# ---------------------------------------------------------
with tab1:
    st.subheader("About the OCT2017 Dataset & Class Proportions")

    col_ds_info, col_ds_chart = st.columns([1.25, 1.0])

    with col_ds_info:
        st.markdown(
            """
            <div class="dataset-card">
                <p style="margin-top:0;">
                    The <strong>OCT2017</strong> dataset consists of optical coherence tomography (OCT) retinal cross-sectional
                    scans categorized into four classes across a training set of around <strong>1 lakh images</strong> and balanced <strong>1,000-image test set</strong>
                    (<strong>250 images per class</strong>, or <strong>25.0%</strong> each):
                </p>
                <ul style="margin-bottom:0; line-height:1.6;">
                    <li><strong>CNV:</strong> Choroidal neovascularization with subretinal fluid/neovascular membrane.</li>
                    <li><strong>DME:</strong> Diabetic macular edema characterized by retinal thickening and intraretinal fluid.</li>
                    <li><strong>DRUSEN:</strong> Multiple drusen deposits beneath the retinal pigment epithelium.</li>
                    <li><strong>NORMAL:</strong> Healthy retina showing preserved foveal contour and retinal layers.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ds_chart:
        test_counts = [250, 250, 250, 250]
        fig_prop, ax_prop = plt.subplots(figsize=(6.8, 3.4), dpi=150)
        fig_prop.patch.set_facecolor("#FAF8F5")
        ax_prop.set_facecolor("#FFFFFF")
        prop_colors = ["#D95D00", "#F59E0B", "#475569", "#0F766E"]
        bars_prop = ax_prop.bar(CLASS_NAMES, test_counts, color=prop_colors, width=0.52)
        ax_prop.set_title("Test Set Class Balance (25.0% Each)", fontsize=10.5, fontweight="bold")
        ax_prop.set_ylabel("Test Scans", fontsize=9)
        ax_prop.set_ylim(0, 300)
        ax_prop.tick_params(axis="both", labelsize=8.5)
        ax_prop.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, cnt in zip(bars_prop, test_counts):
            ax_prop.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 6,
                f"{cnt} (25%)",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
        plt.tight_layout()
        st.pyplot(fig_prop, use_container_width=False)

    st.subheader("Model Comparison Summary (OCT2017)")

    comparison_rows = [
        {
            "Architecture": "Custom CNN Baseline",
            "Test Accuracy": "83.80%",
            "Test Loss": "0.7212",
            "Training Time": "40.0 min",
            "DRUSEN → CNV": "23",
            "Status": "Completed",
        },
        {
            "Architecture": "ResNet18 (Selected)",
            "Test Accuracy": "97.30%",
            "Test Loss": "0.0877",
            "Training Time": "~35.0 min",
            "DRUSEN → CNV": "26",
            "Status": "Selected for App",
        },
        {
            "Architecture": "MobileNetV2",
            "Test Accuracy": "94.50%",
            "Test Loss": "0.1660",
            "Training Time": "41.3 min",
            "DRUSEN → CNV": "47",
            "Status": "Completed",
        },
        {
            "Architecture": "DenseNet121",
            "Test Accuracy": "95.50%",
            "Test Loss": "0.1234",
            "Training Time": "66.3 min",
            "DRUSEN → CNV": "37",
            "Status": "Completed",
        }
    ]
    def render_bold_header_table(rows: list[dict[str, str]]) -> str:
        headers = list(rows[0].keys())
        th_html = "".join(
            f"<th style='text-align:left; padding:8px 12px; border-bottom:2px solid #D95D00; "
            f"background-color:#FFF7ED; color:#1E293B; font-weight:700;'>{h}</th>"
            for h in headers
        )
        body_rows = []
        for r in rows:
            tds = "".join(
                f"<td style='padding:8px 12px; border-bottom:1px solid #E6E0D6; color:#1E293B;'>{r[h]}</td>"
                for h in headers
            )
            body_rows.append(f"<tr>{tds}</tr>")
        return (
            "<div style='overflow-x:auto; background:#FFFFFF; border:1px solid #E6E0D6; border-radius:10px;'>"
            "<table style='width:100%; border-collapse:collapse; font-size:0.92rem;'>"
            f"<thead><tr>{th_html}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table></div>"
        )
    st.markdown(render_bold_header_table(comparison_rows), unsafe_allow_html=True)

    completed_names = ["Custom CNN", "ResNet18", "MobileNetV2", "DenseNet121"]
    accuracies = [83.80, 97.30, 94.50, 95.50]
    losses = [0.7212, 0.0877, 0.1660, 0.1234]
    times_min = [40.0, 35.0, 41.3, 66.3]
    drusen_cnv = [23, 26, 47, 37]
    bar_colors = ["#94A3B8", "#D95D00", "#F59E0B", "#475569"]

    # Row A: 2 compact graphs side-by-side (Accuracy & Loss)
    row_a1, row_a2 = st.columns(2)

    with row_a1:
        fig_acc, ax_acc = plt.subplots(figsize=(5.4, 2.9))
        fig_acc.patch.set_facecolor("#FAF8F5")
        ax_acc.set_facecolor("#FFFFFF")
        bars_acc = ax_acc.bar(completed_names, accuracies, color=bar_colors, width=0.52)
        ax_acc.set_title("Test Accuracy (%)", fontsize=10.5, fontweight="bold")
        ax_acc.set_ylim(75, 101)
        ax_acc.tick_params(axis="both", labelsize=8.5)
        ax_acc.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, v in zip(bars_acc, accuracies):
            ax_acc.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{v:.2f}%",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
        plt.tight_layout()
        st.pyplot(fig_acc, use_container_width=False)

    with row_a2:
        fig_loss, ax_loss = plt.subplots(figsize=(5.4, 2.9))
        fig_loss.patch.set_facecolor("#FAF8F5")
        ax_loss.set_facecolor("#FFFFFF")
        bars_loss = ax_loss.bar(completed_names, losses, color=bar_colors, width=0.52)
        ax_loss.set_title("Test Loss (Lower is Better)", fontsize=10.5, fontweight="bold")
        ax_loss.set_ylim(0, 0.85)
        ax_loss.tick_params(axis="both", labelsize=8.5)
        ax_loss.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, v in zip(bars_loss, losses):
            ax_loss.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{v:.4f}",
                ha="center",
                fontsize=8,
            )
        plt.tight_layout()
        st.pyplot(fig_loss, use_container_width=False)

    # Row B: 2 compact graphs side-by-side (Training Time & DRUSEN -> CNV Misclassifications)
    row_b1, row_b2 = st.columns(2)

    with row_b1:
        fig_time, ax_time = plt.subplots(figsize=(5.4, 2.9))
        fig_time.patch.set_facecolor("#FAF8F5")
        ax_time.set_facecolor("#FFFFFF")
        bars_time = ax_time.bar(completed_names, times_min, color=bar_colors, width=0.52)
        ax_time.set_title("Training Time (Minutes)", fontsize=10.5, fontweight="bold")
        ax_time.set_ylim(0, 78)
        ax_time.tick_params(axis="both", labelsize=8.5)
        ax_time.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, v in zip(bars_time, times_min):
            ax_time.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{v:.1f}m",
                ha="center",
                fontsize=8,
            )
        plt.tight_layout()
        st.pyplot(fig_time, use_container_width=False)

    with row_b2:
        fig_err, ax_err = plt.subplots(figsize=(5.4, 2.9))
        fig_err.patch.set_facecolor("#FAF8F5")
        ax_err.set_facecolor("#FFFFFF")
        bars_err = ax_err.barh(completed_names[::-1], drusen_cnv[::-1], color=bar_colors[::-1], height=0.52)
        ax_err.set_title("DRUSEN → CNV Errors (out of 250)", fontsize=10.5, fontweight="bold")
        ax_err.set_xlabel("Misclassified Count", fontsize=8.5)
        ax_err.set_xlim(0, 56)
        ax_err.tick_params(axis="both", labelsize=8.5)
        ax_err.grid(axis="x", linestyle="--", alpha=0.35)
        for bar, v in zip(bars_err, drusen_cnv[::-1]):
            ax_err.text(
                bar.get_width() + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{v}",
                va="center",
                fontsize=8,
                fontweight="bold",
            )
        plt.tight_layout()
        st.pyplot(fig_err, use_container_width=False)

    st.subheader("Implementation Workflow")
    st.markdown(
        """
        1. **Preprocess OCT Scans:** Resize scans to `224×224` RGB and normalize with ImageNet mean and standard deviation.
        2. **Benchmark Backbones in Colab:** Evaluate Custom CNN, ResNet18, MobileNetV2, and DenseNet121 across accuracy, loss, runtime, and confusion patterns.
        3. **Select Efficient Model:** Choose **ResNet18** (`97.30%` accuracy, `0.0877` loss, `~35.0` min runtime) for fast inference and balanced efficiency.
        4. **Explain & Serve in VS Code:** Load `resnet18_oct2017.pth` in Streamlit and pair predictions with `layer4[-1]` Grad-CAM heatmaps.
        """
    )

# ---------------------------------------------------------
# TAB 2: Upload Scan Image & Display Prediction Results
# ---------------------------------------------------------
with tab2:
    st.subheader("Upload an OCT Scan for Classification")

    uploaded_file = st.file_uploader(
        "Choose an OCT retinal image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        resized_pil = pil_img.resize((224, 224))
        rgb_01 = np.asarray(resized_pil, dtype=np.float32) / 255.0

        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        cam, pred_idx, confidence, probs = cam_engine.generate(input_tensor)
        colored_heatmap = cam_to_colored_heatmap(cam, size=(224, 224))
        overlay_img = make_overlay(rgb_01, cam, alpha=0.4)

        predicted_class = CLASS_NAMES[pred_idx]

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                f"""
                <div class="pred-orange-card">
                <div class="pred-orange-label">Predicted Disease Category</div>
                <div class="pred-orange-value">{predicted_class}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="pred-orange-card">
                    <div class="pred-orange-label">Confidence Score</div>
                    <div class="pred-orange-value">{confidence * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            st.image(resized_pil, caption="Uploaded OCT Scan (224×224)", width=260)
        with col_img2:
            st.image(colored_heatmap, caption="Grad-CAM Heatmap (layer4[-1])", width=260)
        with col_img3:
            st.image(overlay_img, caption=f"Overlay ({predicted_class}: {confidence:.2%})", width=260)

        st.subheader("Class Probability Distribution")
        fig_prob, ax_prob = plt.subplots(figsize=(5.6, 2.9))
        fig_prob.patch.set_facecolor("#FAF8F5")
        ax_prob.set_facecolor("#FFFFFF")
        prob_colors = ["#D95D00" if i == pred_idx else "#64748B" for i in range(len(CLASS_NAMES))]
        bars = ax_prob.bar(CLASS_NAMES, probs * 100.0, color=prob_colors, width=0.52)
        ax_prob.set_ylabel("Probability (%)", fontsize=8.5)
        ax_prob.set_ylim(0, 110)
        ax_prob.tick_params(axis="both", labelsize=8.5)
        ax_prob.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, p in zip(bars, probs):
            ax_prob.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2.0,
                f"{p * 100:.1f}%",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
        plt.tight_layout()
        st.pyplot(fig_prob, use_container_width=False)
    else:
        st.info("Upload an OCT image above to view the predicted category, confidence score, and Grad-CAM visualization.")

# Mandatory Educational Prototype Disclaimer
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>Disclaimer:</strong> This application is an <strong>educational prototype</strong> built for machine learning
        experimentation and visualization. It is <strong>not a clinical diagnostic tool</strong> and must not be used to
        diagnose, treat, cure, or prevent any medical or ophthalmic condition. Always consult a qualified healthcare
        professional or ophthalmologist for clinical evaluation.
    </div>
    """,
    unsafe_allow_html=True,
)

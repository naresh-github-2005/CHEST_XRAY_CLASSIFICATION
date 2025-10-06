# app.py
import io
import os
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# download helpers
import requests

# prefer gdown for Google Drive links (handles large files / confirm token)
try:
    import gdown
    _HAS_GDOWN = True
except Exception:
    _HAS_GDOWN = False

# -------------------------
# Config / constants
# -------------------------
APP_DIR = Path.cwd()
MODEL_LOCAL_NAME = "resnet50_multilabel_best.pth"
LABELS_FILE = APP_DIR / "labels.json"
DEFAULT_THRESHOLD = 0.5
NUM_CLASSES = 15

# -------------------------
# Utilities: model download & load
# -------------------------
def download_via_requests(url: str, dest: Path):
    """Download a file via streaming requests (for public URLs)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
    return dest

def download_from_gdrive(gdrive_url_or_id: str, dest: Path):
    """Download file from Google Drive using gdown.
    Accepts either full share URL or file id.
    """
    if not _HAS_GDOWN:
        raise RuntimeError("gdown is not available in environment. Add gdown to requirements.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # If user gave file id (just 33+ chars), build url
    if "drive.google.com" not in gdrive_url_or_id and len(gdrive_url_or_id) > 20:
        file_id = gdrive_url_or_id.strip()
        url = f"https://drive.google.com/file/d/1zdgE_-yk-SGnSFRDlxzV12QPcCOX0Kux/view?usp=sharing&export=download"
    else:
        url = gdrive_url_or_id
    gdown.download(url, str(dest), quiet=False)
    return dest

@st.cache_resource(show_spinner=True)
def ensure_model_downloaded(model_url: Optional[str], local_path: Path) -> Path:
    """
    Ensure model file exists locally. If not, try to download from model_url.
    model_url precedence:
      1) streamlit secrets: st.secrets["MODEL_URL"]
      2) environment variable MODEL_URL
      3) argument model_url passed here
      4) local file already present
    Returns path to local model file.
    """
    # 1) If already present locally, return
    if local_path.exists():
        return local_path

    # 2) Try secret
    try:
        secret_url = st.secrets.get("MODEL_URL")
    except Exception:
        secret_url = None

    env_url = os.environ.get("MODEL_URL")
    chosen = secret_url or env_url or model_url
    if not chosen:
        raise FileNotFoundError(
            f"No model found at {local_path}. Provide a download URL via Streamlit Secrets (MODEL_URL) "
            "or environment var MODEL_URL or pass model_url to ensure_model_downloaded."
        )

    st.info("Downloading model from provided URL... (this may take a while)")
    start = time.time()
    if "drive.google.com" in chosen or (_HAS_GDOWN and len(chosen) > 20 and "http" not in chosen):
        # prefer gdown for Google Drive or file-id
        if not _HAS_GDOWN:
            raise RuntimeError("Model URL looks like Google Drive but 'gdown' is not installed. Add gdown to requirements.")
        download_from_gdrive(chosen, local_path)
    else:
        download_via_requests(chosen, local_path)
    st.success(f"Downloaded model to {local_path} in {time.time()-start:.1f}s")
    return local_path

# -------------------------
# Model & transforms
# -------------------------
@st.cache_resource(show_spinner=True)
def build_model_and_load_ckpt(ckpt_path: Optional[Path], num_classes: int) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt
        model.load_state_dict(state)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

# -------------------------
# GradCAM helpers
# -------------------------
def get_resnet_target_layer(model):
    try:
        return model.layer4[-1].conv3
    except Exception:
        for m in reversed(list(model.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m
    raise RuntimeError("No Conv2d found")

def compute_gradcam_overlay(model, input_tensor, class_idx, resize_hw=(224,224)):
    # input_tensor: 1,C,H,W on device
    # produce overlay uint8
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m,s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
        std=[1/s for s in [0.229,0.224,0.225]]
    )
    img_unnorm = inv_normalize(input_tensor[0].cpu())
    img_np = img_unnorm.permute(1,2,0).numpy()
    img_np = np.clip(img_np, 0, 1)
    img_for_overlay = np.array(Image.fromarray((img_np*255).astype('uint8')).resize((resize_hw[1], resize_hw[0])))/255.0

    target_layer = get_resnet_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])  # do not pass use_cuda var - version independent
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(img_for_overlay, grayscale_cam, use_rgb=True)
    if hasattr(cam, "clear_hooks"):
        try:
            cam.clear_hooks()
        except Exception:
            pass
    return visualization

# -------------------------
# UI start
# -------------------------
st.set_page_config(page_title="CXR Multilabel (ResNet50) - Deploy", layout="wide")
st.title("CXR Multi-label Inference (ResNet50) - GitHub + Drive deployment")

# Sidebar: let user set model URL (overrides secret)
st.sidebar.header("Model download settings")
st.sidebar.markdown("Provide a **public** URL to your `model.pth` (Google Drive share link or direct URL).")
st.sidebar.markdown("Recommended: add MODEL_URL as a Streamlit Secret in the Cloud app settings.")
typed_url = st.sidebar.text_input("Temporary model URL (paste here to override secret/env)", value="")

# decide model URL priority: typed_url > st.secrets > env
model_url_arg = typed_url.strip() if typed_url.strip() else None

# get local model destination
local_model_path = APP_DIR / MODEL_LOCAL_NAME

# try to ensure model is downloaded (or already present)
try:
    ckpt_path = ensure_model_downloaded(model_url_arg, local_model_path)
except Exception as e:
    st.warning(str(e))
    ckpt_path = local_model_path if local_model_path.exists() else None

# load labels
if LABELS_FILE.exists():
    labels = json.loads(LABELS_FILE.read_text())
else:
    labels = [
        "Aortic_enlargement","Atelectasis","Calcification","Cardiomegaly",
        "Consolidation","ILD","Infiltration","Lung_Opacity","Nodule_Mass",
        "Other_lesion","Pleural_effusion","Pleural_thickening",
        "Pneumothorax","Pulmonary_fibrosis","No_finding"
    ]
if len(labels) != NUM_CLASSES:
    st.warning(f"labels.json length {len(labels)} != NUM_CLASSES ({NUM_CLASSES}). Using default names.")

# load model (cached)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write("Device:", str(device))
with st.spinner("Loading model into memory..."):
    model = build_model_and_load_ckpt(ckpt_path, NUM_CLASSES)
    model = model.to(device)

val_transform = get_val_transform()

# image uploader and processing UI (similar to previous app)
uploaded = st.file_uploader("Upload chest X-ray images (jpg/png)", type=['jpg','jpeg','png'], accept_multiple_files=True)
threshold = st.slider("Prediction threshold", 0.01, 0.99, float(DEFAULT_THRESHOLD), 0.01)

if uploaded:
    results = []
    cols = st.columns(2)
    for u in uploaded:
        pil = Image.open(u).convert("RGB")
        input_tensor = val_transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        # predictions
        pred_idx = [i for i,p in enumerate(probs) if p >= threshold]
        pred_names = [labels[i] for i in pred_idx] if pred_idx else ["No finding >= threshold"]

        # display
        left, right = st.columns([1,1])
        with left:
            st.image(pil, use_column_width=True, caption=u.name)
        with right:
            st.markdown("**Predictions**")
            for i,p in enumerate(probs):
                st.write(f"{labels[i]:<24s} : {p:.3f}")
            st.markdown("**Top predictions**")
            st.write(", ".join(pred_names))

            st.markdown("**Grad-CAM**")
            sel = st.selectbox("Select class to visualize", options=list(range(len(labels))), format_func=lambda x: labels[x], key=u.name)
            if st.button(f"Generate Grad-CAM ({u.name})"):
                try:
                    vis = compute_gradcam_overlay(model, input_tensor, sel, resize_hw=(224,224))
                    st.image(vis, caption=f"Grad-CAM: {labels[sel]}", use_column_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")

        # append results row
        row = {"filename": u.name}
        for i in range(len(labels)):
            row[labels[i]] = float(probs[i])
        results.append(row)

    df = pd.DataFrame(results).set_index("filename")
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv().encode(), file_name="predictions.csv", mime="text/csv")

import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import os
import gdown  # install via requirements
from torchvision.models import resnet50


# -----------------------
# Config
# -----------------------
MODEL_GDRIVE_ID = "1zdgE_-yk-SGnSFRDlxzV12QPcCOX0Kux"  # <-- replace with your file's ID
MODEL_PATH = "resnet50_multilabel_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 15
CLASS_NAMES = [
    "Aortic_enlargement","Atelectasis","Calcification","Cardiomegaly",
    "Consolidation","ILD","Infiltration","Lung_Opacity","Nodule_Mass",
    "Other_lesion","Pleural_effusion","Pleural_thickening",
    "Pneumothorax","Pulmonary_fibrosis","No_finding"
]

# -----------------------
# Download model from Drive if not exists
# -----------------------
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}"
    st.info("Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------
# Load model
# -----------------------

NUM_CLASSES = 15
model = resnet50(weights=None)  # do not load ImageNet weights
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load the checkpoint dictionary
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])  # <-- use 'model_state' key

model.to(DEVICE)
model.eval()


# -----------------------
# Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# -----------------------
# Streamlit UI
# -----------------------
st.title("Chest X-ray: Normal vs Abnormal Detection")
st.write("Upload chest X-ray images and get predictions with confidence scores.")

uploaded_files = st.file_uploader("Upload X-ray images", type=["jpg","png","jpeg"], accept_multiple_files=True)
threshold = st.slider("Confidence threshold", 0.1, 0.99, 0.5, 0.01)

if uploaded_files:
    results = []
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy()[0]

        # Normal vs Abnormal logic
        no_finding_prob = probs[14]
        abnormal_probs = np.delete(probs, 14)
        abnormal_classes = np.array(CLASS_NAMES[:-1])

        if no_finding_prob >= threshold:
            label = "Normal"
            confidence = no_finding_prob
        elif abnormal_probs.max() >= threshold:
            label = "Abnormal"
            confidence = abnormal_probs.max()
        else:
            label = "Normal (low confidence)"
            confidence = no_finding_prob

        # Top 3 abnormal predictions
        top3_idx = abnormal_probs.argsort()[-3:][::-1]
        top3_classes = abnormal_classes[top3_idx]
        top3_probs = abnormal_probs[top3_idx]
        top3_str = ", ".join([f"{c} ({p:.2f})" for c, p in zip(top3_classes, top3_probs)])

        results.append({
            "Filename": file.name,
            "Label": label,
            "Confidence": float(confidence),
            "Top 3 Abnormal Classes": top3_str
        })

    df = pd.DataFrame(results)
    st.dataframe(df)

    # CSV download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")


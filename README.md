# 🩻 Chest X-Ray Abnormality Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

This project focuses on **multi-label classification of chest X-ray images** to detect thoracic abnormalities using deep learning.  
A **ResNet-50** model was trained on the **VinBigData Chest X-ray Abnormalities Detection** dataset and deployed as an interactive **Streamlit** web app.

---

## 🌐 Live Demo

🚀 **Try the live Streamlit app here:**  
👉 [https://chestxrayclassification-hmqvglkkkmgsd6qnqbzrep.streamlit.app/](https://chestxrayclassification-hmqvglkkkmgsd6qnqbzrep.streamlit.app/)

---

## 📦 Dataset

| Type | Description | Link |
|------|--------------|------|
| 🏥 **Original Dataset** | VinBigData Chest X-ray Abnormalities Detection (DICOM format) | [View on Kaggle](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) |
| 🖼️ **Converted JPG Dataset** | Preprocessed and resized images used for model training | [View on Kaggle](https://www.kaggle.com/datasets/nareshr22mis0015/chest-xray) |

---

## 🧠 Model Details

- **Architecture:** ResNet-50 (ImageNet pre-trained)
- **Framework:** PyTorch  
- **Loss Function:** BCEWithLogitsLoss (multi-label)
- **Optimizer:** Adam  
- **Scheduler:** StepLR  
- **Explainability:** Grad-CAM visualization for activation heatmaps

📥 **Download Trained Model:**  
[Model Checkpoint (model.pth)](https://drive.google.com/file/d/1zdgE_-yk-SGnSFRDlxzV12QPcCOX0Kux/view?usp=sharing)

---

## 🧩 Project Files

| File | Description |
|------|--------------|
| `app.py` | Streamlit web app |
| `requirements.txt` | Python dependencies |
| `labels.json` | Class label mappings |
| `model.pth` | Trained ResNet-50 model checkpoint |

---

## ⚙️ Setup & Installation

<details>
<summary>▶️ Step-by-step Setup Guide (click to expand)</summary>

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/chest-xray-classification.git
cd chest-xray-classification

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # (Windows)
# OR
source venv/bin/activate  # (Linux/Mac)

3️⃣ Install Dependencies
pip install -r requirements.txt


💡 Tip: If you have a GPU, install CUDA-compatible PyTorch first.
Check supported versions: PyTorch Installation Guide

4️⃣ Place Required Files

Ensure these are inside your app folder:

app.py
requirements.txt
labels.json
model.pth

5️⃣ Run the Streamlit App
streamlit run app.py


Now open your browser and visit:

http://localhost:8501

</details>
📊 Results & Visualization

Multi-label classification on VinBigData Chest X-rays

Grad-CAM visualizations highlight lung regions influencing model predictions

Achieves strong accuracy and interpretability on unseen images

Example output (in app):

✅ Predicted abnormalities with probabilities

🔥 Grad-CAM heatmap overlay on X-ray

🧠 Tech Stack
Category	Tools
Language	Python
DL Framework	PyTorch
Model Architecture	ResNet-50
Visualization	Grad-CAM, Matplotlib
Frontend	Streamlit
Deployment	Streamlit Community Cloud
Dataset Source	Kaggle
🗂️ Folder Structure
chest-xray-classification/
│
├── app.py
├── requirements.txt
├── labels.json
├── model.pth
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── notebooks/
│   └── train_resnet50.ipynb
└── README.md

🧑‍💻 Author

Naresh R
💼 Deep Learning & Computer Vision Enthusiast
🌐 Streamlit App: https://chestxrayclassification-hmqvglkkkmgsd6qnqbzrep.streamlit.app/

📦 Model: Download from Google Drive

📜 License

This project is open-sourced under the MIT License.
Feel free to use, modify, and share with proper attribution.

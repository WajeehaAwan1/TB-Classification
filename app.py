import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# 🧱 Page setup
st.set_page_config(page_title="TB X-ray Classifier", layout="centered")
st.title("🩺 TB Chest X-ray Classification")
st.write("Upload a chest X-ray image (PNG or JPG) and the model will predict:")

# 🏷️ Class labels
class_names = ['Active Tuberculosis', 'Obsolete Pulmonary Tuberculosis', 'Healthy']

# 🧠 Load model
def load_model():
    try:
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)  # 3 output classes
        model.load_state_dict(torch.load("resnet18_tb_final.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# ✅ Load the model outside the function
model = load_model()

# 🧼 Preprocess uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return transform(image).unsqueeze(0)

# 📤 Upload interface
uploaded_file = st.file_uploader("📤 Upload X-ray image", type=["jpg", "jpeg", "png"])

# 🔍 Predict and display results
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    if model:
        with st.spinner("🔍 Predicting..."):
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).numpy()[0]
                pred_idx = np.argmax(probs)
                pred_label = class_names[pred_idx]

        # ✅ Show result
        st.success(f"🧠 **Prediction:** {pred_label}")
        st.write("### 🔬 Confidence Scores")
        for i, class_name in enumerate(class_names):
            st.write(f"- **{class_name}**: {probs[i]*100:.2f}%")

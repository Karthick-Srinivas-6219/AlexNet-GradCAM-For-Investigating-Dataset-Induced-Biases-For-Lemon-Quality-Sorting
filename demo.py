import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="🍋 GradCAM-Visualizer: For AlexNet based Lemon Quality Grading",
    layout="centered"
)

st.title("🍋 Lemon Quality Classification with Grad-CAM")
st.write("Upload a lemon image to classify quality and visualize model attention.")
st.subheader("Grad-CAM Class Selection")

selected_class = st.selectbox(
    "Choose class to visualize:",
    ["Fresh", "Rotten"]
    )



# -------------------------------------------------
# LOAD MODEL (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = models.alexnet(weights=None)
    model.classifier[6] = nn.Linear(4096, 1)

    checkpoint = torch.load("frozen_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)
    model.eval()

    return model


model = load_model()

print('hello')
# -------------------------------------------------
# TRANSFORMS
# -------------------------------------------------
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Lemon Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # -------------------------------------------------
    # SAFE IMAGE READ (Streamlit-safe)
    # -------------------------------------------------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    np_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    st.image(np_img, caption="Uploaded Image", use_container_width=True)

    # -------------------------------------------------
    # MODEL INPUT
    # -------------------------------------------------
    pil_image = Image.fromarray(np_img)
    input_tensor = test_transforms(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    # -------------------------------------------------
    # PREDICTION DISPLAY
    # -------------------------------------------------
    if prob > 0.5:
        predicted_label = "Fresh 🍋"
        label_color = "green"
    else:
        predicted_label = "Rotten 🟤"
        label_color = "red"

    st.markdown(
        f"<h2 style='color:{label_color}; text-align:center;'>"
        f"{predicted_label}</h2>",
        unsafe_allow_html=True
    )


    # -------------------------------------------------
    # DROPDOWN FOR GRADCAM CLASS
    # -------------------------------------------------


    # -------------------------------------------------
    # PREPARE VISUALIZATION IMAGE (224x224 float32)
    # -------------------------------------------------
    resized_img = cv2.resize(np_img, (224, 224))
    vis_img = resized_img.astype(np.float32) / 255.0

    # -------------------------------------------------
    # GRADCAM SETUP
    # -------------------------------------------------
    target_layers = [model.features[10]]  # last conv layer of AlexNet

    grad_cam = GradCAM(
        model=model,
        target_layers=target_layers
    )
    if selected_class == "Fresh":
        # positive class
        targets = [ BinaryClassifierOutputTarget(1)]
    else:
    # negative class -> invert gradient
        targets = [ BinaryClassifierOutputTarget(0)]

    grayscale_cam = grad_cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    cam_image = show_cam_on_image(
        vis_img,
        grayscale_cam,
        use_rgb=True
    )

    # -------------------------------------------------
    # DISPLAY SIDE BY SIDE
    # -------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(resized_img, use_container_width=True)

    with col2:
        st.subheader(f"Grad-CAM ({selected_class})")
        st.image(cam_image, use_container_width=True)

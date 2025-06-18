# app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
import numpy as np
import pandas as pd
from model import FERModel  # modelo definido aparte

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FERModel()
model.load_state_dict(torch.load("modelo_entrenado.pth", map_location=device))
model.eval().to(device)

# Emociones
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# Colores para cada emoción (pueden cambiarse)
emotion_colors = {
    'Angry': '#E63946',
    'Disgust': '#6A994E',
    'Fear': '#9A8C98',
    'Happy': '#F4D35E',
    'Sad': '#457B9D',
    'Surprise': '#F9844A',
    'Neutral': '#A8A7A7'
}

# Detector de caras
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)

# Transformaciones
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# --- Streamlit UI ---

# Encabezado con estilo y emojis
st.markdown(
    """
    <h1 style='text-align:center; color:#3B4252; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;'>
        🎭 Detector de emociones faciales 😄😠😢
    </h1>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Subí una foto grupal (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen original", use_column_width=True)

    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        st.warning("No se detectaron caras. 🤔 Intentá con otra foto.")
    else:
        st.success(f"Se detectaron {len(boxes)} cara(s) 👀")

        draw = ImageDraw.Draw(image)
        # Fuente para texto (ajustá el path si usás Colab u otro entorno)
        try:
            font = ImageFont.truetype("arial.ttf", size=22)
        except:
            font = ImageFont.load_default()

        for i, box in enumerate(boxes):
            face = image.crop(box).convert("L").resize((48, 48))
            input

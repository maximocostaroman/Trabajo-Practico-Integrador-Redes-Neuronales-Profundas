# app.py
import streamlit as st
from PIL import Image, ImageDraw
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

# Detector de caras
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)

# Transformaciones
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Streamlit UI
st.title("Detector de emociones faciales 😄😠😢")
uploaded_file = st.file_uploader("Subí una foto grupal", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen original", use_column_width=True)

    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        st.warning("No se detectaron caras.")
    else:
        draw = ImageDraw.Draw(image)

        for i, box in enumerate(boxes):
            face = image.crop(box).convert("L").resize((48, 48))
            input_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                pred = torch.argmax(probs).item()
                emotion = class_names[pred]
                topk = torch.topk(probs, 3)
                top_emotions = [(class_names[i], float(p) * 100) for i, p in zip(topk.indices, topk.values)]

            # Dibujo en la imagen
            draw.rectangle(box.tolist(), outline="red", width=2)
            text = f"{emotion} ({top_emotions[0][1]:.1f}%)"
            draw.text((box[0], box[1] - 10), text, fill="red")

            # Mostrar tabla con top-3 emociones en Streamlit
            st.write(f"🧠 **Predicción para rostro #{i+1}**: {emotion}")
            st.table(pd.DataFrame(top_emotions, columns=["Emoción", "Confianza (%)"]))

        st.image(image, caption="Emociones detectadas", use_column_width=True)

# app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
import pandas as pd
from model import FERModel  # modelo definido aparte

# Configuración del dispositivo y modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FERModel()
model.load_state_dict(torch.load("modelo_entrenado.pth", map_location=device))
model.eval().to(device)

# Emociones y colores asociados
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': '#E63946',
    'Disgust': '#6A994E',
    'Fear': '#9A8C98',
    'Happy': '#F4D35E',
    'Sad': '#457B9D',
    'Surprise': '#F9844A',
    'Neutral': '#A8A7A7'
}

# Detector de caras MTCNN
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)

# Transformaciones para el modelo
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Interfaz Streamlit
st.markdown(
    """
    <h1 style='text-align:center; color:#3B4252; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;'>
        🎭 Detector de emociones faciales 😄😠😢
    </h1>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Subí una foto grupal (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen original", use_container_width=True)

    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        st.warning("No se detectaron caras. 🤔 Intentá con otra foto.")
    else:
        st.success(f"Se detectaron {len(boxes)} cara(s) 👀")

        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", size=22)
        except:
            font = ImageFont.load_default()

        for i, box in enumerate(boxes):
            face = image.crop(box).convert("L").resize((48, 48))
            input_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                pred = torch.argmax(probs).item()
                emotion = class_names[pred]
                topk = torch.topk(probs, 3)
                top_emotions = [(class_names[idx], float(p) * 100) for idx, p in zip(topk.indices, topk.values)]

            color = emotion_colors.get(emotion, "red")

            draw.rectangle(box.tolist(), outline=color, width=3)
            text = f"{emotion} ({top_emotions[0][1]:.1f}%)"
            draw.text((box[0], box[1] - 28), text, fill=color, font=font)

            st.markdown(f"### 🧠 Predicción para rostro #{i+1}: <span style='color:{color};'>{emotion}</span>", unsafe_allow_html=True)

            df_emotions = pd.DataFrame(top_emotions, columns=["Emoción", "Confianza (%)"])
            df_emotions["Confianza (%)"] = df_emotions["Confianza (%)"].map(lambda x: f"{x:.1f}%")
            st.table(df_emotions)

        st.image(image, caption="Emociones detectadas 🎉", use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>💡 Pro tip: ¡Sonreí para que te detecte 'Happy'! 😄</p>", unsafe_allow_html=True)

import threading, time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title="UNAH-Campus Comayagua", page_icon="🙂", layout="centered")

IMG = 128          # tamaño de entrada de la CNN
MARGEN = 0.20      # margen alrededor del rostro, igual que en el entrenamiento
UMBRAL = 0.5       # probabilidad de mujer a partir de la cual se dice "Mujer"
SUAVIZADO = 6      # predicciones que se promedian por cara
DIST_MAX = 100     # píxeles para considerar que es la misma cara entre cuadros
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

NOMBRE = "Hjasher Medina"
NUMERO_CUENTA = "20231900178"


@st.cache_resource
def recursos():
    from tensorflow import keras
    modelo = keras.models.load_model("mejor_modelo.keras", compile=False)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return {"modelo": modelo, "detector": detector, "clahe": clahe,
            "lock": threading.Lock(), "caras": []}


R = recursos()


def preparar_rostro(frame_bgr, caja):
    """Recorta el rostro con margen, pasa de BGR a RGB y lo deja en 128x128 (0-255),
    exactamente como lo vio la red durante el entrenamiento."""
    x, y, w, h = caja
    mx, my = int(w * MARGEN), int(h * MARGEN)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(frame_bgr.shape[1], x + w + mx), min(frame_bgr.shape[0], y + h + my)
    rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    return rgb.astype("float32")


def analizar(img_bgr):
    """Detecta rostros y devuelve sus cajas y la probabilidad de mujer de cada uno."""
    gris = R["clahe"].apply(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))   # la mejora de luz solo ayuda a detectar
    cajas = R["detector"].detectMultiScale(gris, 1.1, 5, minSize=(90, 90))
    if len(cajas) == 0:
        return [], []
    lote = np.stack([preparar_rostro(img_bgr, c) for c in cajas])
    with R["lock"]:
        probs = R["modelo"](lote, training=False).numpy().flatten()
    return cajas, probs


def dibujar(img, caja, p_mujer):
    es_mujer = p_mujer >= UMBRAL
    conf = p_mujer if es_mujer else 1 - p_mujer
    etiqueta = f"{'Mujer' if es_mujer else 'Hombre'} ({conf:.0%})"
    color = (180, 60, 255) if es_mujer else (255, 140, 0)   # BGR, rosa y azul
    x, y, w, h = map(int, caja)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    cv2.putText(img, etiqueta, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def procesar(frame):
    """Se ejecuta con cada cuadro de la cámara en vivo."""
    img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)   # efecto espejo
    cajas, probs = analizar(img)
    ahora = time.time()

    with R["lock"]:
        caras = R["caras"]
        caras[:] = [c for c in caras if ahora - c["t"] < 1.5]   # olvidar caras que ya no están
        for c in caras:
            c["usada"] = False
        for caja, p in zip(cajas, probs):
            x, y, w, h = caja
            centro = (x + w / 2, y + h / 2)
            elegida, mejor_d = None, DIST_MAX
            for c in caras:
                if c["usada"]:
                    continue
                d = np.hypot(centro[0] - c["centro"][0], centro[1] - c["centro"][1])
                if d < mejor_d:
                    elegida, mejor_d = c, d
            if elegida is None:
                elegida = {"hist": deque(maxlen=SUAVIZADO)}
                caras.append(elegida)
            elegida.update(centro=centro, t=ahora, usada=True)
            elegida["hist"].append(float(p))
            dibujar(img, caja, sum(elegida["hist"]) / len(elegida["hist"]))

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- Interfaz ----------------
st.title("UNAH-Campus Comayagua")
st.markdown(f"**{NOMBRE}** · Número de cuenta {NUMERO_CUENTA}")
st.subheader("🙂 Detector Hombre / Mujer con CNN")
st.caption("CNN entrenada con el dataset UTKFace, 92% de precisión en el grupo de prueba.")

tab_vivo, tab_foto = st.tabs(["📹 Cámara en vivo", "📷 Foto o imagen"])

with tab_vivo:
    st.write("Presiona START y permite el acceso a la cámara.")
    webrtc_streamer(
        key="genero",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=procesar,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tab_foto:
    foto = st.camera_input("Toma una foto")
    archivo = st.file_uploader("O sube una imagen", type=["jpg", "jpeg", "png"])
    fuente = foto if foto is not None else archivo
    if fuente is not None:
        img = cv2.imdecode(np.frombuffer(fuente.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img.shape[1] > 960:   # reducir fotos muy grandes
            esc = 960 / img.shape[1]
            img = cv2.resize(img, None, fx=esc, fy=esc)
        cajas, probs = analizar(img)
        for caja, p in zip(cajas, probs):
            dibujar(img, caja, float(p))
        st.image(img, channels="BGR")
        if len(cajas) == 0:
            st.warning("No se detectó ningún rostro. Prueba con la cara de frente y buena luz.")

st.caption("Limitaciones, el modelo se apoya mucho en rasgos como el cabello, y con poca luz o la cara de lado puede equivocarse.")
import os
import threading
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from tensorflow import keras

# ---------------- Datos del encabezado ----------------
NOMBRE = "Hjasher Medina"
NUMERO_CUENTA = "20231900178"

# ---------------- Parámetros ----------------
IMG = 128          # tamaño de entrada de la red
MARGEN = 0.20      # margen extra alrededor del rostro
UMBRAL = 0.5       # prob. de "mujer" a partir de la cual se dice Mujer
SUAVIZADO = 6      # cuadros que se promedian por rostro
DIST_MAX = 100     # distancia máx. (px) para seguir al mismo rostro

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

st.set_page_config(page_title="UNAH-Campus Comayagua", page_icon="🎓", layout="centered")


# ---------------- TURN (Metered) ----------------
def obtener_ice_servers():
    usuario = os.environ.get("METERED_USER")
    clave = os.environ.get("METERED_PASS")
    if not usuario or not clave:
        return RTC_CONFIGURATION["iceServers"]
    return [
        {"urls": "stun:stun.relay.metered.ca:80"},
        {"urls": "turn:global.relay.metered.ca:80", "username": usuario, "credential": clave},
        {"urls": "turn:global.relay.metered.ca:80?transport=tcp", "username": usuario, "credential": clave},
        {"urls": "turn:global.relay.metered.ca:443", "username": usuario, "credential": clave},
        {"urls": "turns:global.relay.metered.ca:443?transport=tcp", "username": usuario, "credential": clave},
    ]


# ---------------- Recursos (se cargan una sola vez) ----------------
@st.cache_resource
def recursos():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mejor_modelo.keras")
    modelo = keras.models.load_model(ruta, compile=False)
    cascada = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return {
        "modelo": modelo,
        "cascada": cascada,
        "clahe": clahe,
        "lock": threading.Lock(),
        "caras": [],  # rostros seguidos entre cuadros
    }


R = recursos()


# ---------------- Lógica ----------------
def detectar_caras(img_bgr):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gris = R["clahe"].apply(gris)  # CLAHE solo para detectar
    cajas = R["cascada"].detectMultiScale(
        gris, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return [tuple(int(v) for v in c) for c in cajas]


def preparar_rostro(img_bgr, caja):
    x, y, w, h = caja
    m = int(MARGEN * max(w, h))
    alto, ancho = img_bgr.shape[:2]
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, ancho), min(y + h + m, alto)
    rostro = img_bgr[y0:y1, x0:x1]
    rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
    rostro = cv2.resize(rostro, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    return rostro.astype("float32")  # 0-255, la red ya reescala por dentro


def predecir(img_bgr, cajas):
    lote = np.stack([preparar_rostro(img_bgr, c) for c in cajas])
    salida = R["modelo"](lote, training=False).numpy().flatten()
    return [float(p) for p in salida]  # probabilidad de "mujer"


def suavizar(cajas, probs):
    """Promedia la probabilidad de cada rostro con sus últimos cuadros."""
    usados = set()
    nuevas, resultado = [], []
    for (x, y, w, h), p in zip(cajas, probs):
        cx, cy = x + w / 2, y + h / 2
        mejor, mejor_d = None, DIST_MAX
        for i, t in enumerate(R["caras"]):
            if i in usados:
                continue
            d = float(np.hypot(cx - t["cx"], cy - t["cy"]))
            if d < mejor_d:
                mejor, mejor_d = i, d
        if mejor is None:
            hist = deque(maxlen=SUAVIZADO)
        else:
            usados.add(mejor)
            hist = R["caras"][mejor]["hist"]
        hist.append(p)
        nuevas.append({"cx": cx, "cy": cy, "hist": hist})
        resultado.append(float(np.mean(hist)))
    R["caras"] = nuevas
    return resultado


def dibujar(img_bgr, cajas, probs):
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, w, h), p in zip(cajas, probs):
        es_mujer = p >= UMBRAL
        etiqueta = "Mujer" if es_mujer else "Hombre"
        confianza = p if es_mujer else 1 - p
        color = (180, 60, 255) if es_mujer else (255, 140, 0)  # BGR
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
        texto = f"{etiqueta} {confianza * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(texto, fuente, 0.8, 2)
        y0 = max(y - th - 12, 0)
        cv2.rectangle(img_bgr, (x, y0), (x + tw + 10, y0 + th + 12), color, -1)
        cv2.putText(img_bgr, texto, (x + 5, y0 + th + 4), fuente, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return img_bgr


def procesar(frame):
    """Se llama por cada cuadro de la cámara en vivo."""
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)  # efecto espejo
    with R["lock"]:
        cajas = detectar_caras(img)
        if cajas:
            probs = suavizar(cajas, predecir(img, cajas))
            img = dibujar(img, cajas, probs)
        else:
            R["caras"] = []
    return av.VideoFrame.from_ndarray(img, format="bgr24")


def analizar_foto(imagen_pil):
    rgb = np.array(imagen_pil.convert("RGB"))
    if rgb.shape[1] > 1280:  # reducir fotos muy grandes
        escala = 1280 / rgb.shape[1]
        rgb = cv2.resize(rgb, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    with R["lock"]:
        cajas = detectar_caras(bgr)
        probs = predecir(bgr, cajas) if cajas else []
    salida = dibujar(bgr.copy(), cajas, probs)
    return cv2.cvtColor(salida, cv2.COLOR_BGR2RGB), probs


# ---------------- Interfaz ----------------
st.title("UNAH-Campus Comayagua")
st.markdown(f"**Estudiante:** {NOMBRE}  \n**Número de cuenta:** {NUMERO_CUENTA}")
st.subheader("Detección de género (Hombre / Mujer) con una red neuronal convolucional")

tab_vivo, tab_foto = st.tabs(["📹 Cámara en vivo", "📷 Foto o imagen"])

with tab_vivo:
    st.write("Presiona START y permite el acceso a la cámara.")
    webrtc_streamer(
        key="genero",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=procesar,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": obtener_ice_servers()},
    )

with tab_foto:
    st.write("Toma una foto con la cámara o sube una imagen.")
    foto = st.camera_input("Tomar foto")
    archivo = st.file_uploader("O sube una imagen", type=["jpg", "jpeg", "png"])
    fuente_img = foto or archivo
    if fuente_img is not None:
        resultado, probs = analizar_foto(Image.open(fuente_img))
        st.image(resultado, use_container_width=True)
        if not probs:
            st.warning("No se detectó ningún rostro. Prueba con más luz y de frente.")
        for i, p in enumerate(probs, 1):
            etiqueta = "Mujer" if p >= UMBRAL else "Hombre"
            conf = p if p >= UMBRAL else 1 - p
            st.write(f"Rostro {i}: **{etiqueta}** ({conf * 100:.0f}% de confianza)")

st.caption(
    "Limitaciones: el modelo se entrenó con UTKFace (~92% de precisión en pruebas), "
    "funciona mejor con rostros de frente y buena luz, y puede equivocarse con "
    "peinados poco comunes, por ejemplo el cabello recogido."
)

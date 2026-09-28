# Modelos de Prueba OpenVINO & Pipeline de Detección EPP

Este repositorio contiene las pruebas, modelos y pipeline de inferencia acelerada con **Intel OpenVINO** para el sistema **GuardVision** y el motor de reglas **VORTEX**.

---

## 📋 Contenido del Repositorio

- **`vortex_stream.py`**: Pipeline integral de visión por computadora.
  - Inferencia optimizada con OpenVINO (CPU con afinidad a P-Cores o iGPU Intel FP16 nativo).
  - Tracking multi-objeto persistente mediante **ByteTrack** (`track_id` estable).
  - Normalización de coordenadas a resolución de planta industrial (1920x1080) compatible con la matriz de homografía y zonas de riesgo de VORTEX.
  - Publicación de telemetría y detecciones en tiempo real mediante socket **ZeroMQ PUSH**.
  - Servidor de streaming de video **MJPEG** HTTP integrado (puerto `8088`) para visualización en el dashboard web de GuardVision sin latencia ni retardo de buffer.
- **`ppe-11s.pt` / `ppe-11s_openvino_model/`**:
  - Modelo YOLO especializado en EPP exportado a formato OpenVINO IR FP16 (`.xml` + `.bin`).
  - Clases normalizadas: `person`, `helmet`, `no_helmet`, `vest`, `no_vest`.
- **`videos/`**:
  - Videos de muestra de entorno industrial para pruebas de inferencia y validación de reglas de seguridad (`10.mp4`, `9.mp4`).
- **`requirements.txt`**:
  - Lista de dependencias del entorno de ejecución.

---

## 🚀 Requisitos e Instalación

Se recomienda utilizar Python 3.10+ o `uv`:

```bash
# Con pip
pip install -r requirements.txt

# O utilizando uv
uv pip install -r requirements.txt
```

### Dependencias Principales
- `ultralytics`: Inferencia YOLO y algoritmos de tracking.
- `openvino`: Motor de ejecución y aceleración por hardware de Intel.
- `opencv-python`: Decodificación de video, escalado y compresión MJPEG.
- `pyzmq`: Comunicación ZeroMQ de baja latencia con VORTEX.

---

## 💻 Uso

### 1. Inferencia sobre video de prueba (con streaming a GuardVision y VORTEX)
```bash
python vortex_stream.py --model ppe-11s_openvino_model --source videos/10.mp4 --fps 15
```

### 2. Uso con cámara web en vivo
```bash
python vortex_stream.py --model ppe-11s_openvino_model --source 0
```

### 3. Parámetros Principales
| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `--model` | `ppe-11s_openvino_model` | Ruta al modelo `.pt` o directorio OpenVINO IR |
| `--source` / `--rtsp` | `0` | Ruta al archivo de video o índice de webcam / URL RTSP |
| `--device` | `AUTO` | Dispositivo OpenVINO (`CPU`, `GPU` o `AUTO`) |
| `--fps` | `15` | Control de cadencia de cuadros (pacing) para evitar acumulación en cola |
| `--conf` | `0.25` | Umbral mínimo de confianza de detección |
| `--mjpeg-port` | `8088` | Puerto del servidor MJPEG para el frontend |
| `--zmq-endpoint`| `tcp://localhost:5555` | Dirección del socket ZeroMQ de VORTEX |
| `--show` | `False` | Muestra ventana gráfica local OpenCV con detecciones |

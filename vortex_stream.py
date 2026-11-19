import argparse
import json
import logging
import os
import sys
import threading
import time
import openvino as ov

# Crear directorio de caché para compilar instantáneamente en futuras ejecuciones
os.makedirs("model_cache", exist_ok=True)

original_compile_model = ov.Core.compile_model

# Monkey-patch para configurar OpenVINO al máximo rendimiento en CPU o GPU (mantenido del main.py original)
def optimized_compile_model(self, model, device_name="CPU", config=None, *args, **kwargs):
    if config is None:
        config = {}

    # Habilitar caché de compilación en disco para reducir tiempos de carga y optimizar shaders
    config["CACHE_DIR"] = "model_cache"

    dev = device_name.upper()
    if "CPU" in dev:
        # Usar únicamente los núcleos de alto rendimiento (P-cores)
        config["SCHEDULING_CORE_TYPE"] = "PCORE_ONLY"
        # Fijar los hilos a los procesadores físicos para evitar cambios de contexto
        config["ENABLE_CPU_PINNING"] = "YES"
        # Forzar precisión bfloat16 si está soportada por hardware (DL Boost) para máxima velocidad
        config["INFERENCE_PRECISION_HINT"] = "bf16"
    elif "GPU" in dev:
        # Optimización para GPU integrada Intel (UHD/Iris Xe)
        config["PERFORMANCE_HINT"] = "LATENCY"
        # Forzar precisión FP16 nativa en la GPU (el hardware de la GPU Intel procesa FP16 al doble de velocidad que FP32)
        config["INFERENCE_PRECISION_HINT"] = "f16"

    return original_compile_model(self, model, device_name, config, *args, **kwargs)

ov.Core.compile_model = optimized_compile_model

import cv2
import http.server
import numpy as np
import socket
import socketserver
from ultralytics import YOLO
import zmq


# El ultralytics del venv de YOLOE (8.4.215) marca 'intel:cpu'/'intel:gpu' como obsoletos
# y lo avisa en cada cuadro. Se siguen usando porque el venv de ppe-11s (8.4.203) solo
# entiende ese formato; el filtro evita llenar el log del pipeline con el mismo aviso.
class _IntelDeviceWarningFilter(logging.Filter):
    def filter(self, record):
        return "'device=intel:" not in record.getMessage()


logging.getLogger("ultralytics").addFilter(_IntelDeviceWarningFilter())


# Clases de YOLOE por defecto: el texto que se le da al modelo, no la clase que ve VORTEX.
YOLOE_DEFAULT_CLASSES = "person,face mask,white lab coat"
YOLOE_DEFAULT_DERIVE = "mask,coat"

# Texto de la clase (YOLOE) -> clase canónica que entiende VORTEX (VORTEX-93). Con
# ppe-11s ningún nombre cambia: "person" es el único que aparece y queda igual.
CANONICAL_CLASSES = {
    "person": "person",
    "face mask": "mask",
    "white lab coat": "coat",
    "lab coat": "coat",
}

# Base del track_id de cada falta deducida: se suma al id de la persona. ByteTrack
# numera desde 1 y en una corrida no llega a millones de pistas, así que el id derivado
# no choca con uno real y es estable mientras la persona conserve el suyo.
DERIVED_TRACK_BASE = {
    "mask": 2_000_000,
    "coat": 3_000_000,
}

# Una prenda pertenece a una persona si al menos esta fracción de su caja cae dentro de
# la caja de la persona (el mismo criterio que ADR-16).
GARMENT_OVERLAP_MIN = 0.5


# Clase para leer la cámara en un hilo independiente (adaptada para soportar RTSP y webcams locales)
class ThreadedCamera:
    def __init__(self, src=0, loop=True):
        self.src = src
        self.loop = loop
        self.is_file = isinstance(src, str) and os.path.isfile(src)
        self.is_url = isinstance(src, str) and (
            src.startswith("rtsp://")
            or src.startswith("rtsps://")
            or src.startswith("http://")
            or src.startswith("https://")
        )

        if self.is_url:
            # Configurar FFmpeg para latencia ultra baja y conexión estable sobre TCP
            # Se evita probesize/analyzeduration extremos que rompen la negociación de buffers H.264 en OpenCV
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"
                "fflags;nobuffer|"
                "flags;low_delay|"
                "max_delay;500000"
            )
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.is_file:
            # Archivo de video local (.mp4, .avi, etc.)
            self.cap = cv2.VideoCapture(src)
        else:
            # Para webcam local usamos CAP_V4L2
            idx = int(src) if isinstance(src, str) and src.isdigit() else src
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

        # Para fuentes en vivo (RTSP o webcam), usamos un hilo de fondo para vaciar buffers.
        # Para archivos locales, leemos frame por frame de forma determinista para reproducir a la velocidad indicada.
        if not self.is_file:
            self.t = threading.Thread(target=self.update, args=())
            self.t.daemon = True
        else:
            self.t = None

    def start(self):
        if self.t is not None:
            self.t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            # Drenar frames de la red en tiempo real tan pronto como llegan
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        if self.is_file:
            with self.lock:
                ret, frame = self.cap.read()
                if not ret:
                    if self.loop:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    else:
                        return False, None
                self.ret = ret
                self.frame = frame
                return ret, frame.copy() if frame is not None else None

        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def release(self):
        self.stopped = True
        if self.t is not None and self.t.is_alive():
            self.t.join(timeout=1.0)
        self.cap.release()


# Buffer compartido y servidor HTTP MJPEG para alimentar el frontend (guardvision-front)
class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.frame_id = 0
        self.condition = threading.Condition()

    def set_frame(self, frame_bytes):
        with self.condition:
            self.frame = frame_bytes
            self.frame_id += 1
            self.condition.notify_all()

    def get_latest(self, last_id):
        with self.condition:
            # Si el cliente ya vio este frame, esperar al siguiente
            while self.frame_id == last_id:
                if not self.condition.wait(timeout=0.1):
                    return self.frame, self.frame_id
            return self.frame, self.frame_id


streaming_output = StreamingOutput()


class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silenciar logs HTTP en consola
        pass

    def do_GET(self):
        if self.path in ("/stream", "/video_feed", "/"):
            # Enviar paquetes TCP inmediatamente sin esperar buffer Nagle ni inflar buffer de envío
            try:
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            except Exception:
                pass

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                last_frame_id = -1
                while True:
                    frame, current_id = streaming_output.get_latest(last_frame_id)
                    if frame is not None and current_id != last_frame_id:
                        last_frame_id = current_id
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()  # Enviar inmediatamente al navegador
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_mjpeg_server(host="0.0.0.0", port=8088):
    try:
        server = ThreadedHTTPServer((host, port), MJPEGHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"Servidor MJPEG para frontend iniciado en: http://localhost:{port}/stream")
        return server
    except Exception as e:
        print(f"Aviso: no se pudo iniciar el servidor MJPEG en el puerto {port}: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detector EPP con OpenVINO + YOLOv8/v11 y salida ZeroMQ hacia VORTEX"
    )
    parser.add_argument(
        "--rtsp",
        "--source",
        dest="rtsp",
        type=str,
        default="0",
        help="Fuente de video: URL RTSP, archivo de video (.mp4/.avi) o '0' para webcam local (default: '0')",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="No reiniciar en bucle cuando la fuente sea un archivo de video local",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PPE2_best.pt",
        help="Ruta al modelo YOLO/OpenVINO (default: 'PPE2_best.pt')",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Umbral de confianza mínimo de detección (default: 0.4)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostrar ventana gráfica con visualización de video y tracking",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="cam_industrial_01",
        help="Identificador de la cámara para la telemetría enviada a VORTEX (default: 'cam_industrial_01')",
    )
    parser.add_argument(
        "--zmq-endpoint",
        type=str,
        default="tcp://*:5557",
        help="Endpoint ZeroMQ PUSH (default: 'tcp://*:5557' para bind, o 'tcp://localhost:5557' para connect)",
    )
    parser.add_argument(
        "--zmq-mode",
        type=str,
        default="bind",
        choices=["bind", "connect"],
        help="Modo del socket ZeroMQ: 'bind' (VORTEX se conecta) o 'connect' (default: 'bind')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Dispositivo de hardware OpenVINO: 'auto', 'cpu' o 'gpu' (default: 'auto')",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Límite de tasa de procesamiento en frames por segundo (default: 15.0, usar 0 para ilimitado)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de entrada de la inferencia en píxeles (default: 640). Un modelo OpenVINO "
        "exportado con otro tamaño usa el suyo",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Solo YOLOE: clases en texto, separadas por coma "
        f"(default con YOLOE: '{YOLOE_DEFAULT_CLASSES}'). En un modelo OpenVINO exportado "
        "las clases ya vienen fijas y esta opción no las cambia",
    )
    parser.add_argument(
        "--derive-missing",
        type=str,
        default=None,
        help="Prendas cuya falta se deduce por persona, separadas por coma (mask, coat). "
        f"Default: '{YOLOE_DEFAULT_DERIVE}' con YOLOE, ninguna con otros modelos; "
        "'none' la desactiva",
    )
    parser.add_argument(
        "--mjpeg-port",
        type=int,
        default=8088,
        help="Puerto HTTP para transmitir video MJPEG a guardvision-front (default: 8088, usar 0 para desactivar)",
    )
    return parser.parse_args()


def is_yoloe(model_path: str) -> bool:
    return "yoloe" in os.path.basename(os.path.normpath(model_path)).lower()


def exported_metadata(model_path: str) -> dict:
    """Lee el metadata.yaml de un modelo OpenVINO exportado ({} si no hay)."""
    meta_path = os.path.join(model_path, "metadata.yaml")
    if not os.path.isfile(meta_path):
        return {}
    # El lector de YAML de ultralytics: según la versión, PyYAML no viene instalado.
    from ultralytics.utils import YAML

    return YAML.load(meta_path) or {}


def load_yoloe_pytorch(model_path: str, classes: list):
    """YOLOE en PyTorch, con las clases fijadas en el arranque.

    Carga el codificador de texto (~254 MB) para convertir las clases en embeddings.
    Es el camino de respaldo si la exportación a OpenVINO falla, y el que permite probar
    otras clases sin reexportar.
    """
    from ultralytics import YOLOE

    print(f"Cargando YOLOE en PyTorch desde: {model_path} con clases {classes}")
    model = YOLOE(model_path)
    model.set_classes(classes)
    return model


def canonical_class(raw_name: str) -> str:
    name = raw_name.lower().strip()
    if name in CANONICAL_CLASSES:
        return CANONICAL_CLASSES[name]
    return name.replace("-", "_").replace(" ", "_")


def garment_belongs(garment_bbox, person_bbox) -> bool:
    gx1, gy1, gx2, gy2 = garment_bbox
    px1, py1, px2, py2 = person_bbox
    garment_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    if garment_area <= 0:
        return False
    inter_w = max(0.0, min(gx2, px2) - max(gx1, px1))
    inter_h = max(0.0, min(gy2, py2) - max(gy1, py1))
    return inter_w * inter_h >= GARMENT_OVERLAP_MIN * garment_area


def derive_missing(detections: list, garments: list, derived_class_ids: dict) -> list:
    """Por cada persona sin la prenda, una detección no_<prenda> con su caja.

    YOLOE detecta lo que hay, no lo que falta: la falta se deduce cuadro a cuadro a partir
    de las personas y las prendas que se ven.
    """
    persons = [d for d in detections if d["class_name"] == "person"]
    derived = []
    for person in persons:
        for garment in garments:
            worn = any(
                d["class_name"] == garment and garment_belongs(d["bbox"], person["bbox"])
                for d in detections
            )
            if worn:
                continue
            derived.append(
                {
                    "track_id": DERIVED_TRACK_BASE[garment] + person["track_id"],
                    "class_id": derived_class_ids[garment],
                    "class_name": f"no_{garment}",
                    "confidence": person["confidence"],
                    "bbox": list(person["bbox"]),
                }
            )
    return derived


def load_openvino_model(model_path: str):
    """
    Carga el modelo asegurando el uso del backend OpenVINO.
    Si se proporciona un checkpoint PyTorch (.pt), se exporta automáticamente a OpenVINO IR
    si aún no existe el directorio compilado correspondiente.
    """
    if model_path.endswith(".pt"):
        ov_model_dir = os.path.splitext(model_path)[0] + "_openvino_model"
        if os.path.exists(ov_model_dir):
            print(f"Cargando modelo OpenVINO existente desde: {ov_model_dir}")
            return YOLO(ov_model_dir, task="detect")
        elif os.path.exists(model_path):
            print(f"Exportando '{model_path}' a formato OpenVINO para máxima aceleración...")
            base_model = YOLO(model_path)
            ov_model_dir = base_model.export(format="openvino", half=True)
            print(f"Exportación completada en: {ov_model_dir}")
            return YOLO(ov_model_dir, task="detect")
        else:
            raise FileNotFoundError(f"No se encontró el archivo de modelo: {model_path}")
    else:
        # Un modelo de segmentación (YOLOE -seg) leído como "detect" interpreta los
        # coeficientes de máscara como clases y devuelve basura: la tarea sale del
        # metadata.yaml de la exportación. Sin metadata, "detect" como siempre.
        task = exported_metadata(model_path).get("task", "detect")
        return YOLO(model_path, task=task)


def main():
    args = parse_args()

    # 1. Configurar ZeroMQ (Socket PUSH hacia VORTEX)
    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUSH)
    # HWM y timeout para evitar bloqueo indefinido si el consumidor no responde
    zmq_socket.setsockopt(zmq.SNDHWM, 100)
    zmq_socket.setsockopt(zmq.SNDTIMEO, 500)

    endpoint = args.zmq_endpoint
    if args.zmq_mode == "bind":
        if "localhost" in endpoint:
            endpoint = endpoint.replace("localhost", "*")
        zmq_socket.bind(endpoint)
        print(f"ZeroMQ PUSH escuchando (bind) en: {endpoint} [esperando conexión de VORTEX]")
    else:
        zmq_socket.connect(endpoint)
        print(f"ZeroMQ PUSH conectado a: {endpoint}")

    # 2. Iniciar servidor HTTP MJPEG para el frontend si está habilitado
    mjpeg_server = None
    if args.mjpeg_port > 0:
        mjpeg_server = start_mjpeg_server(port=args.mjpeg_port)

    # 2. Detectar hardware óptimo para OpenVINO
    core = ov.Core()
    if args.device == "auto":
        hw = "gpu" if "GPU" in core.available_devices else "cpu"
    else:
        hw = args.device.lower()

    yoloe = is_yoloe(args.model)
    # Un .pt de YOLOE corre en PyTorch: exportarlo solo, como se hace con los demás .pt,
    # perdería las clases. La exportación con clases fijas es export_yoloe.py.
    pytorch_backend = yoloe and args.model.endswith(".pt")

    if pytorch_backend:
        if hw == "gpu":
            print("Aviso: YOLOE en PyTorch corre solo en CPU (torch CPU); --device gpu es para OpenVINO.")
        device_arg = "cpu"
        device_name = "CPU (PyTorch)"
    else:
        # En Ultralytics los dispositivos OpenVINO se direccionan como 'intel:cpu' o 'intel:gpu'
        device_arg = f"intel:{hw}"
        device_name = hw.upper()
    print(f"Hardware seleccionado: {device_name} ({device_arg})")

    # 3. Cargar el modelo
    if args.classes is not None and not yoloe:
        print("Aviso: --classes es solo para YOLOE; se ignora con este modelo.")
    if pytorch_backend:
        classes_text = args.classes if args.classes is not None else YOLOE_DEFAULT_CLASSES
        classes = [c.strip() for c in classes_text.split(",") if c.strip()]
        model = load_yoloe_pytorch(args.model, classes)
        imgsz = args.imgsz
    else:
        if yoloe and args.classes is not None:
            print("Aviso: el modelo OpenVINO trae las clases fijas de la exportación; --classes no las cambia.")
        model = load_openvino_model(args.model)
        ov_dir = args.model if os.path.isdir(args.model) else os.path.splitext(args.model)[0] + "_openvino_model"
        # La exportación es de tamaño fijo: otro --imgsz no correría.
        meta_imgsz = exported_metadata(ov_dir).get("imgsz")
        imgsz = args.imgsz
        if meta_imgsz and int(meta_imgsz[0]) != args.imgsz:
            imgsz = int(meta_imgsz[0])
            print(f"Aviso: {ov_dir} está exportado a {imgsz}; se usa {imgsz} en vez de --imgsz {args.imgsz}.")
    print(f"Tamaño de entrada: {imgsz}")

    # 4. Determinar fuente de captura
    src = int(args.rtsp) if args.rtsp.isdigit() else args.rtsp
    print(f"Iniciando captura de video en: {args.rtsp}...")
    cam = ThreadedCamera(src, loop=not args.no_loop).start()

    # Esperar hasta 5 segundos a que la cámara capture el primer frame
    t_start = time.time()
    first_frame = None
    first_ret = False
    while time.time() - t_start < 5.0:
        first_ret, first_frame = cam.read()
        if first_ret and first_frame is not None:
            break
        time.sleep(0.1)

    if not first_ret or first_frame is None:
        print(f"Error: No se pudo recibir señal de video desde '{args.rtsp}'.")
        cam.release()
        zmq_socket.close()
        zmq_context.term()
        sys.exit(1)

    # 5. Precalentamiento del modelo para compilar kernels y evitar latencia inicial
    print(f"Precalentando modelo en {device_name} ({device_arg})...")
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model.track(
        source=dummy_frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        device=device_arg,
        imgsz=imgsz,
    )

    # 6. Faltas deducidas por persona. Se valida después del precalentamiento, cuando el
    # modelo ya está cargado en el dispositivo elegido y model.names es el suyo.
    if args.derive_missing is not None:
        derive_text = args.derive_missing
    else:
        derive_text = YOLOE_DEFAULT_DERIVE if yoloe else "none"
    garments = []
    if derive_text.strip().lower() != "none":
        garments = [g.strip().lower() for g in derive_text.split(",") if g.strip()]
    derived_class_ids = {}
    if garments:
        unknown = [g for g in garments if g not in DERIVED_TRACK_BASE]
        if unknown:
            print(f"Error: --derive-missing no conoce {unknown}; opciones: {sorted(DERIVED_TRACK_BASE)}.")
            sys.exit(2)
        model_classes = {canonical_class(n) for n in model.names.values()}
        # Sin la clase de la prenda (o sin personas) en el modelo, cada persona saldría
        # siempre sin ella: mejor no arrancar.
        absent = [g for g in ["person", *garments] if g not in model_classes]
        if absent:
            print(f"Error: el modelo no detecta {absent} (clases: {sorted(model_classes)}); no se puede deducir la falta.")
            sys.exit(2)
        # class_id propio para cada falta, a continuación de los del modelo, para que no
        # se confunda con una clase real.
        next_id = max(model.names) + 1
        derived_class_ids = {g: next_id + i for i, g in enumerate(garments)}
        print(f"Falta deducida por persona: {', '.join('no_' + g for g in garments)}")

    print("Transmisión de video establecida. Enviando telemetría a VORTEX...")
    if args.show:
        print("Ventana gráfica activa. Presiona 'q' en la ventana para salir.")
    else:
        print("Ejecutando en modo headless (sin ventana). Presiona Ctrl+C para detener.")

    frame_id = 0
    prev_time = time.time()
    target_interval = (1.0 / args.fps) if args.fps > 0 else 0
    fps = args.fps if args.fps > 0 else 30.0

    try:
        while True:
            loop_start = time.time()
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_id += 1
            h, w = frame.shape[:2]
            timestamp_ms = int(time.time() * 1000)

            # Inferencia y tracking con ByteTrack (persist=True mantiene track_id entre frames)
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                verbose=False,
                device=device_arg,
                imgsz=imgsz,
            )

            detections = []
            scale_x = 1920.0 / w if w > 0 else 1.0
            scale_y = 1080.0 / h if h > 0 else 1.0

            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                r = results[0]
                boxes = r.boxes
                for box_idx, box in enumerate(boxes):
                    # Bounding box escalado a 1920x1080 para coincidir con la homografía de VORTEX
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = [
                        round(float(x1 * scale_x), 1),
                        round(float(y1 * scale_y), 1),
                        round(float(x2 * scale_x), 1),
                        round(float(y2 * scale_y), 1),
                    ]

                    # Clase y confianza (VORTEX parsea class_name en minúsculas: "person", "helmet", "vest", etc.)
                    cls_id = int(box.cls.item())
                    raw_name = r.names.get(cls_id, str(cls_id)) if hasattr(r, "names") else str(cls_id)
                    cls_name = canonical_class(raw_name)
                    conf = round(float(box.conf.item()), 2)

                    # ID de tracking asignado por ByteTrack (VORTEX requiere u64 obligatorio)
                    if box.id is not None:
                        track_id = int(box.id.item())
                    else:
                        track_id = int((cls_id + 1) * 1000 + box_idx + 1)
                    if track_id < 0:
                        continue

                    detections.append(
                        {
                            "track_id": track_id,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": bbox,
                        }
                    )

            if garments:
                detections.extend(derive_missing(detections, garments, derived_class_ids))

            # Estructura JSON exacta requerida por VORTEX (en espacio calibrado 1920x1080)
            payload = {
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "camera_id": args.camera_id,
                "frame_width": 1920,
                "frame_height": 1080,
                "detections": detections,
            }

            # Enviar mensaje por ZeroMQ
            try:
                zmq_socket.send_string(json.dumps(payload))
            except zmq.ZMQError:
                # Si el socket expira o buffer está lleno, se continúa sin interrumpir el stream
                pass

            # Transmitir frame de la cámara hacia el frontend (guardvision-front)
            if mjpeg_server is not None:
                # Si la resolución es alta (ej. 1080p), reescalar para el visor web.
                # Reducir el payload JPEG de 250KB a ~40-50KB previene que el navegador (Chrome)
                # acumule frames en su buffer TCP interno y genere 1 a 2 segundos de lag visual.
                if w > 1280:
                    preview_frame = cv2.resize(frame, (1280, int(h * (1280.0 / w))), interpolation=cv2.INTER_LINEAR)
                else:
                    preview_frame = frame
                ret_enc, jpeg_buf = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if ret_enc:
                    streaming_output.set_frame(jpeg_buf.tobytes())

            # Control de tasa de frames (pacing para respetar args.fps, ej. 15 FPS)
            if target_interval > 0:
                elapsed = time.time() - loop_start
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Cálculo de FPS suavizado (promedio móvil exponencial)
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = 0.85 * fps + 0.15 * instant_fps

            # Visualización opcional
            if args.show:
                annotated_frame = results[0].plot() if len(results) > 0 else frame

                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f} | Hardware: {device_name} | Tracks: {len(detections)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("VORTEX - Detector EPP (Q para salir)", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Salida solicitada por el usuario.")
                    break
            else:
                if frame_id % 30 == 0:
                    print(
                        f"[Frame {frame_id}] Detecciones: {len(detections)} | FPS: {fps:.1f} | Hardware: {device_name}"
                    )

    except KeyboardInterrupt:
        print("\nInterrupción por teclado detectada.")
    finally:
        if mjpeg_server is not None:
            mjpeg_server.shutdown()
            mjpeg_server.server_close()
        cam.release()
        zmq_socket.close()
        zmq_context.term()
        if args.show:
            cv2.destroyAllWindows()
        print("Proceso finalizado correctamente.")


if __name__ == "__main__":
    main()

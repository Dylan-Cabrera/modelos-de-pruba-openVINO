import openvino as ov
import os

# Crear directorio de caché para compilar instantáneamente en futuras ejecuciones
os.makedirs("model_cache", exist_ok=True)

original_compile_model = ov.Core.compile_model

# Monkey-patch para configurar OpenVINO al máximo rendimiento en CPU o GPU
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

from ultralytics import YOLO
import cv2
import sys
import time
import threading

# Clase para leer la cámara en un hilo independiente (evita retrasos y esperas de E/S)
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        # Ajustamos a una resolución base estándar en la cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.ret, self.frame = self.cap.read()
        if self.ret and self.frame is not None:
            # Pre-redimensionar al tamaño del modelo en el hilo de fondo
            self.frame = cv2.resize(self.frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        self.stopped = False
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # El redimensionamiento se hace en segundo plano para dejar libre la CPU del hilo principal
                resized_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
                self.ret = ret
                self.frame = resized_frame
            time.sleep(0.002)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

# Hilo dedicado únicamente a la inferencia (libera al hilo principal de la carga de procesar el modelo)
class InferenceWorker:
    def __init__(self, model_path, camera, device):
        self.model = YOLO(model_path)
        self.camera = camera
        self.device = device
        self.stopped = False
        
        self.annotated_frame = None
        self.latency_ms = 0.0
        self.new_frame_available = False
        
        self.t = threading.Thread(target=self.run, args=())
        self.t.daemon = True

    def start(self):
        self.t.start()
        return self

    def run(self):
        while not self.stopped:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            
            t0 = time.time()
            
            # Ejecutar predicción optimizada
            results = self.model(
                frame,
                stream=True,
                half=True,
                imgsz=640,
                conf=0.35,   # Subimos el umbral para descartar falsos positivos rápido
                iou=0.5,     # NMS más rápido
                max_det=10,  # Reducido a 10 detecciones máximas para acelerar post-procesamiento
                classes=[0], # Solo personas
                verbose=False,
                device=f"intel:{self.device.lower()}"
            )
            
            temp_frame = frame
            for r in results:
                # Restauramos las etiquetas de texto de detección con labels=True
                temp_frame = r.plot(labels=True, conf=True)
                
            self.latency_ms = (time.time() - t0) * 1000
            self.annotated_frame = temp_frame
            self.new_frame_available = True
            time.sleep(0.001)

    def stop(self):
        self.stopped = True
        self.t.join()

# 1. Intentar iniciar la cámara con hilos
print("Iniciando cámara en segundo plano...")
cam = ThreadedCamera(0).start()
time.sleep(1.0)  # Esperar a que la cámara capture el primer frame

ret, frame = cam.read()
if not ret or frame is None:
    print("Error: No se pudo abrir la webcam (índice 0).")
    cam.release()
    sys.exit(1)

# 2. Detectar automáticamente el mejor hardware disponible
core = ov.Core()
device_name = "GPU" if "GPU" in core.available_devices else "CPU"
print(f"Cargando modelo YOLO (OpenVINO) en hardware: {device_name}...")

# 3. Arrancar el trabajador de inferencia en segundo plano
worker = InferenceWorker("personas-seg_best_openvino_model/", cam, device_name).start()

print("Iniciando transmisión. Presiona 'q' para salir.")
prev_time = time.time()

while True:
    if worker.new_frame_available and worker.annotated_frame is not None:
        frame = worker.annotated_frame
        worker.new_frame_available = False
        latency = worker.latency_ms

        # Calcular FPS de renderizado de la interfaz
        curr_time = time.time()
        time_diff = curr_time - prev_time
        prev_time = curr_time
        fps = 1.0 / time_diff if time_diff > 0 else 0

        # Mostrar estadísticas de rendimiento en tiempo real
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Latencia: {latency:.1f}ms [{device_name}]",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # MOSTRAR LA IMAGEN
        cv2.imshow("YOLO Segmentacion - Presiona 'q' para salir", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos de forma limpia
worker.stop()
cam.release()
cv2.destroyAllWindows()
print("Webcam cerrada correctamente.")


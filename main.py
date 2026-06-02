from ultralytics import YOLO
import cv2

# Cargar tu modelo de segmentación entrenado
model = YOLO("personas-seg_best.pt")

model.export(format="openvino")

ov_model = YOLO("personas-seg_best_openvino_model/")


# Abrir video (0 para la webcam integrada)
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error al abrir la webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo recibir el frame de la cámara")
        break

    # Ejecutar la predicción (usamos stream=True para optimizar el video)
    results = ov_model(frame, stream=True)
    
    # Inicializamos una variable por si acaso no hay detecciones
    annotated_frame = frame 

    for r in results:
        # r.plot() dibuja automáticamente las cajas Y LAS MÁSCARAS de segmentación
        annotated_frame = r.plot() 

    # MOSTRAR LA IMAGEN (Fuera del bucle 'for')
    cv2.imshow("YOLO Segmentacion - Presiona 'q' para salir", annotated_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(67) & 0xFF == ord('q'):
        break

# Liberar recursos de forma limpia
cap.release()
cv2.destroyAllWindows()
print("Webcam cerrada correctamente.")
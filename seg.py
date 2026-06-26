import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, AutoProcessor

# 1. Cargar la imagen
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

# 2. Cargar procesador y modelo
processor = AutoProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", device_map="auto")

# 3. Preparar las entradas
inputs = processor(images=image, return_tensors="pt").to(model.device)

# 4. Inferencia (usamos torch.no_grad() para no consumir memoria extra)
with torch.no_grad():
    outputs = model(**inputs)

# 5. Post-procesar los logits para obtener el mapa de segmentación con el tamaño original
# PIL devuelve (ancho, alto), pero el procesador espera (alto, ancho)
target_sizes = [image.size[::-1]] 
predicted_map = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]

# 6. Mover a CPU y convertir a NumPy
segmentation_map = predicted_map.cpu().numpy()

# 7. Guardar la imagen con un mapa de colores para poder visualizar las clases
plt.imsave("resultado_segmentacion.png", segmentation_map, cmap="viridis")

print("Imagen guardada correctamente como 'resultado_segmentacion.png'")
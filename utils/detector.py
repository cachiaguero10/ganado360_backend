from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw

# Carga el modelo YOLOv8 (descarga automática si no existe)
model = YOLO("yolov8n.pt")

def contar_vacas(pil_image):
    """Detecta vacas en la imagen y devuelve cantidad + imagen marcada"""
    results = model.predict(pil_image, conf=0.35, verbose=False)

    # Dibujar las detecciones
    image = pil_image.copy()
    draw = ImageDraw.Draw(image)

    count = 0
    for r in results[0].boxes:
        cls_id = int(r.cls[0])
        label = model.names[cls_id]
        if label.lower() in ["cow", "cattle", "bull"]:
            count += 1
            x1, y1, x2, y2 = r.xyxy[0].tolist()
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1, y1 - 10), "Vaca", fill="white")

    return count, image
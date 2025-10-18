import torch
from PIL import Image
import io

# Cargar modelo YOLOv5 (liviano)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

async def detectar_vacas(file):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Procesar la imagen
    results = model(image)
    detecciones = results.pandas().xyxy[0]

    # Contar solo las vacas detectadas
    vacas = len(detecciones[detecciones['name'] == 'cow'])
    return {
        "total_vacas": int(vacas),
        "detecciones": detecciones.to_dict(orient="records")
    }
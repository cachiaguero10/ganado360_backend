from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import torch
import io
from PIL import Image
import os
import uuid

app = FastAPI()

# 📂 Carpeta para salidas
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📥 Descargar automáticamente el modelo si no existe
MODEL_PATH = "yolov8n.pt"
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo YOLOv8n...")
    torch.hub.download_url_to_file(
        'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        MODEL_PATH
    )

# 🚀 Cargar el modelo YOLOv8
print("Cargando modelo...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
print("Modelo listo.")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        results = model(image)
        detections = results.pandas().xyxy[0]
        count = len(detections)

        output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.jpg")
        results.save(save_dir=OUTPUT_DIR)
        marked_img = os.path.join(OUTPUT_DIR, os.listdir(OUTPUT_DIR)[0])
        os.rename(marked_img, output_path)

        return JSONResponse(content={
            "count": int(count),
            "image_url": f"/get_image/{os.path.basename(output_path)}"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get_image/{filename}")
async def get_image(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    return JSONResponse(content={"error": "Archivo no encontrado"}, status_code=404)

@app.get("/")
def home():
    return {"status": "Servidor activo - Ganado360"}
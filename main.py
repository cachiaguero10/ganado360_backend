from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io

app = FastAPI()

# Cargar modelo YOLOv8
MODEL_PATH = "yolov8n.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        results = model(image)
        detections = results.pandas().xyxy[0]
        count = len(detections)
        return JSONResponse(content={"count": count})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def home():
    return {"status": "Servidor activo - Ganado360"}
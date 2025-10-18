from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from utils.detector import contar_vacas
from io import BytesIO
import base64
from PIL import Image

app = FastAPI(title="Ganado360 Backend YOLOv8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Servidor Ganado360 con YOLOv8 activo ✅"}

@app.post("/count")
async def count(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    count, result_image = contar_vacas(image)

    buffer = BytesIO()
    result_image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"vacas_detectadas": count, "imagen_resultado": img_str}
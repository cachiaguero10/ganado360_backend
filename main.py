from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import base64

# Inicialización de FastAPI
app = FastAPI(title="Ganado360 Backend", description="API para conteo de vacas con YOLOv8", version="1.0")

# Cargar el modelo YOLOv8 (ligero y rápido)
RUTA_DEL_MODELO = "yolov8n.pt"
modelo = YOLO(RUTA_DEL_MODELO)


@app.get("/")
def root():
    return {"status": "Servidor activo", "modelo": "YOLOv8n listo"}


@app.post("/contar/")
async def contar_vacas(file: UploadFile = File(...)):
    try:
        # Leer imagen subida
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Ejecutar detección
        resultados = modelo(image)

        # Contar cuántas detecciones hay de tipo 'cow' (vaca)
        conteo_vacas = 0
        for r in resultados:
            if r.names:
                for clase_id in r.boxes.cls:
                    nombre_clase = r.names[int(clase_id)]
                    if nombre_clase.lower() == "cow":
                        conteo_vacas += 1

        # Convertir imagen resultante a base64 para enviar marcada
        img_result = resultados[0].plot()  # imagen con cajas
        img_pil = Image.fromarray(img_result)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse({
            "resultado": "ok",
            "total_vacas": conteo_vacas,
            "imagen_marcada": img_base64
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
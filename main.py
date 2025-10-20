from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch.serialization import add_safe_globals
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from PIL import Image
import io
import os
import time

# === Seguridad PyTorch ===
add_safe_globals([tasks.DetectionModel])

# === Inicialización ===
app = FastAPI(title="Ganado360 - Detección de Vacas", version="2.0")

# === Permitir acceso desde Flutter / Web ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Cargar modelo YOLOv8s ===
os.makedirs("modelos", exist_ok=True)
RUTA_DEL_MODELO = os.path.join("modelos", "yolov8s.pt")
print(f"Cargando modelo desde: {RUTA_DEL_MODELO}")

try:
    if not os.path.exists(RUTA_DEL_MODELO):
        print("Descargando modelo YOLOv8s...")
        modelo = YOLO("yolov8s.pt")  # descarga oficial
        modelo.save(RUTA_DEL_MODELO)
    else:
        modelo = YOLO(RUTA_DEL_MODELO)
    print("✅ Modelo YOLOv8s cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    modelo = None

# === Ruta principal ===
@app.get("/")
def root():
    return {"estado": "servidor activo", "modelo": "Ganado360 v8s"}

# === Endpoint para procesar imagen ===
@app.post("/contar_vacas/")
async def contar_vacas(file: UploadFile = File(...)):
    inicio = time.time()
    try:
        if modelo is None:
            return JSONResponse(content={"error": "Modelo no cargado"}, status_code=500)

        contenido = await file.read()
        imagen = Image.open(io.BytesIO(contenido))

        # Ejecutar detección con confianza mínima baja
        resultados = modelo.predict(imagen, conf=0.15, imgsz=640, verbose=False)

        # Contar vacas detectadas
        conteo = 0
        for r in resultados:
            for clase in r.boxes.cls.tolist():
                nombre = r.names[int(clase)]
                if nombre.lower() in ["cow", "vaca", "cattle"]:
                    conteo += 1

        duracion = round(time.time() - inicio, 2)
        print(f"✅ Detección completada en {duracion}s - Vacas: {conteo}")

        return JSONResponse(
            content={
                "vacas_detectadas": conteo,
                "tiempo_procesamiento": f"{duracion}s",
            }
        )

    except Exception as e:
        print(f"⚠ Error procesando imagen: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
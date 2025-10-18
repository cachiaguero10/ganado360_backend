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

# === Configuración de seguridad para PyTorch ===
add_safe_globals([tasks.DetectionModel])

# === Inicialización de FastAPI ===
app = FastAPI(title="Ganado360 - Detección de Vacas", version="1.0")

# === Permitir acceso desde cualquier origen (Flutter o Web) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Cargar modelo YOLO ===
os.makedirs("modelos", exist_ok=True)  # crea la carpeta si no existe
RUTA_DEL_MODELO = os.path.join("modelos", "yolov8n.pt")
print(f"Cargando modelo desde: {RUTA_DEL_MODELO}")

try:
    # si el modelo no está, se descarga automáticamente
    if not os.path.exists(RUTA_DEL_MODELO):
        print("Descargando modelo YOLOv8n...")
        modelo = YOLO("yolov8n.pt")  # descarga el modelo base
        modelo.save(RUTA_DEL_MODELO)
    else:
        modelo = YOLO(RUTA_DEL_MODELO)

    print("✅ Modelo YOLO cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")

# === Ruta principal ===
@app.get("/")
def root():
    return {"estado": "servidor activo", "modelo": "Ganado360"}

# === Endpoint para procesar imagen ===
@app.post("/contar_vacas/")
async def contar_vacas(file: UploadFile = File(...)):
    try:
        # Leer la imagen recibida
        contenido = await file.read()
        imagen = Image.open(io.BytesIO(contenido))

        # Ejecutar detección
        resultados = modelo(imagen)

        # Contar detecciones de vacas
        conteo = 0
        for r in resultados:
            nombres = r.names
            clases_detectadas = r.boxes.cls.tolist()
            for clase in clases_detectadas:
                nombre = nombres[int(clase)]
                if nombre.lower() in ["cow", "vaca", "cattle"]:
                    conteo += 1

        # Responder con el conteo
        return JSONResponse(content={"vacas_detectadas": conteo})

    except Exception as e:
        print(f"⚠ Error procesando imagen: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
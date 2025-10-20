from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torch.serialization import add_safe_globals
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from PIL import Image
import io
import os

# === Configuración segura de PyTorch ===
add_safe_globals([tasks.DetectionModel])

# === Inicialización de la API ===
app = FastAPI(title="Ganado360 - Conteo de Ganado YOLOv8s", version="1.0")

# === CORS (permite conexión desde app Flutter o Web) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Configurar modelo YOLOv8s ===
os.makedirs("modelos", exist_ok=True)
RUTA_DEL_MODELO = os.path.join("modelos", "yolov8s.pt")

print(f"Cargando modelo desde: {RUTA_DEL_MODELO}")

try:
    if not os.path.exists(RUTA_DEL_MODELO):
        print("Descargando modelo YOLOv8s...")
        modelo = YOLO("yolov8s.pt")  # descarga automática
    else:
        modelo = YOLO(RUTA_DEL_MODELO)

    print("✅ Modelo YOLOv8s cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")

# === Endpoint raíz ===
@app.get("/")
def root():
    return {"estado": "servidor activo", "modelo": "Ganado360 YOLOv8s"}

# === Endpoint para contar vacas ===
@app.post("/contar_vacas/")
async def contar_vacas(file: UploadFile = File(...)):
    try:
        contenido = await file.read()
        imagen = Image.open(io.BytesIO(contenido))

        resultados = modelo.predict(source=imagen, conf=0.25, iou=0.45, verbose=False)

        conteo = 0
        for r in resultados:
            nombres = r.names
            clases_detectadas = r.boxes.cls.tolist()
            for clase in clases_detectadas:
                nombre = nombres[int(clase)]
                if nombre.lower() in ["cow", "vaca", "cattle"]:
                    conteo += 1

        return JSONResponse(content={"vacas_detectadas": conteo})

    except Exception as e:
        print(f"⚠ Error procesando imagen: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
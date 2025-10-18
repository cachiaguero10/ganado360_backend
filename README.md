# Ganado360 Backend YOLOv8

Servicio FastAPI que detecta vacas con YOLOv8.

## Despliegue en Render
1. Crear cuenta en https://render.com
2. Nuevo servicio Web → “New + Web Service”
3. Subir este ZIP o conectar tu repo GitHub
4. Configurar:
   - Runtime: Python 3.10
   - Start Command:
     uvicorn main:app --host 0.0.0.0 --port 10000
   - Free Tier
5. Esperar a que Render termine el deploy.
6. URL final → Ejemplo:
   https://ganado360-backend.onrender.com/count
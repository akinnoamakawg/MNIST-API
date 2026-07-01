import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from redis import Redis

app = FastAPI(title="API Batching MNIST")
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_conn = Redis(host=redis_host, port=6379)

@app.post("/predict", status_code=202)
async def prediction_batch(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Format d'image invalide")
    
    image_bytes = await file.read()

    # Générer un identifiant unique pour la tâche
    task_id = str(uuid.uuid4())

    # Stocker l'image dans Redis sous forme de JSON
    task_data = {
        "task_id": task_id,
        "image_hex": image_bytes.hex()  # Convertir les octets en hexadécimal pour le stockage
    }
    # LPUSH ajoute l'élément au début de la liste Redis "images_queue"
    redis_conn.lpush("images_queue", json.dumps(task_data))

    # On initialise le statut de la tâche dans Redis
    redis_conn.set(f"result:{task_id}", json.dumps({"statut": "en attente"}), ex=3600)

    return {
        "task_id": task_id,
        "statut": "en attente de traitement."
    }

@app.get("/predict/{task_id}")
def get_prediction_batch(task_id: str):
    result_raw = redis_conn.get(f"result:{task_id}")
    if not result_raw:
        raise HTTPException(status_code=404, detail="ID de tâche introuvable.")
    
    return json.loads(result_raw.decode("utf-8"))
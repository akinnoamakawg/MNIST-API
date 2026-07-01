import os
import time
import torch
import io
import json
from PIL import Image
from torchvision import transforms
from train import MNISTCNN
from redis import Redis

# Connexion Redis et Modele
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_conn = Redis(host=redis_host, port=6379)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MNISTCNN().to(device)
model.load_state_dict(torch.load("modele/mnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir en niveaux de gris
    transforms.Resize((28, 28)),                   # Redimensionner à 28x28
    transforms.ToTensor(),                          # Convertir en tenseur
    transforms.Normalize((0.1307,), (0.3081,))     # Normaliser les valeurs entre -1 et 1
])

BATCH_SIZE = 32  # Taille du batch pour le traitement par lot
MAX_WAIT_TIME = 0.05  # 50 illisecondes maximum d'attente pour le batch

print("Worker RQ prêt à traiter les tâches...")

while True:
    batch_tasks = []
    start_time = time.time()

    # On essaie de recuperer des tâches de la file d'attente jusqu'à ce qu'on atteigne la taille du batch ou le temps max
    while len(batch_tasks) < BATCH_SIZE:
        # On check s'il y a un truc dans la queue (pour respecter le temps max d'attente)
        # On utilise RPOP pour recuperer le plus ancien element de la liste Redis (FIFO)
        task_raw = redis_conn.rpop("images_queue")

        if task_raw:
            # task_raw contient du JSON avec {"task_id": "...", "image_hex": "..."}
            task_data = json.loads(task_raw.decode("utf-8"))
            batch_tasks.append(task_data)
        else:
            # S'il n'y a plus rien dans la queue, on fait une micro-pause pour ne pas surcharger le CPU
            time.sleep(0.001)

        # Si le temps d'attente max est depasse, on sort de la boucle pour traiter ce qu'on a
        if time.time() - start_time > MAX_WAIT_TIME:
            break
    
    if not batch_tasks:
        # Si la queue est completement vide, on fait une pause avant de reessayer
        time.sleep(0.01)
        continue

    # On prepare les images pour le batch
    tensors_list = []
    for task in batch_tasks:
        # On recupere les octets de l'image (stockes en hex dans le JSON)
        img_bytes = bytes.fromhex(task["image_hex"])
        image = Image.open(io.BytesIO(img_bytes))
        tensor = transform(image).to(device)
        tensors_list.append(tensor)

    # On empile toutes les images du batch en un seul tenseur : ex [8,1,28,28] pour un batch de 8 images
    batch_tensor = torch.stack(tensors_list)

    # Inference ultra rapide en une seule fois !
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)

    # On sauvegarde les resultats dans Redis pour chaque tache du batch
    for i, task in enumerate(batch_tasks):
        result = {
            "statut": "succès",
            "prediction": int(predictions[i].item()),
            "confiance": float(confidences[i].item())
        }
        # On stocke le resultat dans une clef Redis unique pendant 1 heure (3600 secondes)
        redis_conn.set(f"result:{task['task_id']}", json.dumps(result), ex=3600)
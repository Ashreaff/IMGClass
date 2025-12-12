import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms

app = FastAPI()

# Charger le modèle
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Charger les noms des classes depuis imagenet_classes.txt
with open("imagenet_classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Définir les transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire et transformer l'image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        # Faire la prédiction
        with torch.no_grad():
            out = model(batch_t)

        # Calculer les probabilités
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        probs, indices = torch.topk(probabilities, 5)

        # Associer les noms des classes aux probabilités
        predictions = {
            class_names[idx]: float(prob)
            for idx, prob in zip(indices.tolist(), probs.tolist())
        }
        return predictions
    except Exception as e:
        return {"error": str(e)}

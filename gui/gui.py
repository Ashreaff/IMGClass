import gradio as gr
import requests
import io
from PIL import Image

def predict(image):
    if image is None:
        return {"Error": 1.0}

    try:
        # Convertir l'image en bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
       
        # Envoyer la requête à l'API
        response = requests.post(
            "http://inference:8000/predict/",
            files={"file": ("image.png", img_byte_arr, "image/png")}
        )
       
        # Vérifier et retourner la réponse
        if response.status_code == 200:
            try:
                predictions = response.json()
                return predictions
            except ValueError:
                return {"Error": "Invalid response format"}
        else:
            return {"Error": f"API Error: {response.status_code}"}
           
    except requests.exceptions.RequestException as e:
        return {"Error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"Error": f"An error occurred: {str(e)}"}

# Créer l'interface Gradio
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    live=True,
    title="Classification d'images",
    description="Téléchargez une image pour la classifier"
)

# Lancer l'application
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)

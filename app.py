
import os
import json
import torch
import gradio as gr

from PIL import Image
from torchvision import models, transforms


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")

# The model will be downloaded/copied here during deployment.
MODEL_PATH = os.path.join(BASE_DIR, "reproduced_A0_resnet50.pth")


# ============================================================
# Load configuration
# ============================================================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


DISEASE_LABELS = config["disease_labels"]

NUM_CLASSES = config["num_classes"]

INPUT_SIZE = tuple(config["input_size"])

MEAN = config["normalization_mean"]

STD = config["normalization_std"]

THRESHOLD = config["threshold"]


# ============================================================
# Device
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ============================================================
# Model
# ============================================================

def create_model():

    model = models.resnet50(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Handle the checkpoint created during the Colab experiment.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)

    model.eval()

    return model


print("=" * 70)
print("RFMiD RETINAL AI APPLICATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Model path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model checkpoint not found: "
        + MODEL_PATH
    )

model = create_model()

print("✓ ResNet-50 model loaded successfully")
print("=" * 70)


# ============================================================
# Prediction function
# ============================================================

def predict(image):

    if image is None:
        return {
            "Error": 0.0
        }

    # Convert uploaded image to RGB.
    image = image.convert("RGB")

    # Apply the same preprocessing used during A0.
    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(DEVICE)

    # Inference only.
    with torch.no_grad():

        logits = model(input_tensor)

        probabilities = torch.sigmoid(logits)

    probabilities = probabilities[0].cpu().numpy()

    # Create dictionary for all diseases.
    results = {
        DISEASE_LABELS[i]: float(probabilities[i])
        for i in range(NUM_CLASSES)
    }

    # Sort by probability.
    results = dict(
        sorted(
            results.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # Convert to percentage for display.
    results_percentage = {
        label: round(probability * 100, 2)
        for label, probability in results.items()
    }

    return results_percentage


# ============================================================
# Interface
# ============================================================

description = """
### RFMiD Multi-Disease Retinal AI

Upload a retinal fundus image to obtain the model's predicted
probabilities for 45 retinal disease categories.

**Model:** ResNet-50  
**Input:** Retinal fundus image  
**Task:** Multi-label classification  
**Dataset:** RFMiD  

### Important

This is a **research and educational prototype**.

The predictions are model outputs and **must not be used for
medical diagnosis, treatment, or clinical decision-making**.
"""


examples_text = """
**How to use:**

1. Upload a retinal fundus image.
2. Click **Analyze Image**.
3. Review the predicted probabilities.
4. Higher percentages indicate higher model confidence,
   not clinical certainty.
"""


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload Retinal Fundus Image"
    ),
    outputs=gr.Label(
        num_top_classes=10,
        label="Model Predictions"
    ),
    title="RFMiD Multi-Disease Retinal AI",
    description=description + examples_text,
    flagging_mode="never"
)


# ============================================================
# Launch
# ============================================================

PORT = int(os.environ.get("PORT", "10000"))

demo.launch(
    server_name="0.0.0.0",
    server_port=PORT
)


import os
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

from huggingface_hub import hf_hub_download


# ============================================
# Configuration
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "model_config.json"
)

HF_REPO_ID = "Kashish-goel185/retinal-ai-resnet50"

MODEL_FILENAME = "reproduced_A0_resnet50.pth"

DEVICE = torch.device("cpu")


# ============================================
# Load configuration
# ============================================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DISEASE_LABELS = config["disease_labels"]
NUM_CLASSES = config["num_classes"]


# ============================================
# Download model from Hugging Face
# ============================================

print("Downloading/loading model from Hugging Face...")

MODEL_PATH = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=MODEL_FILENAME
)

print(f"Model available at: {MODEL_PATH}")


# ============================================
# Create ResNet-50
# ============================================

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


# ============================================
# Load checkpoint
# ============================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("ResNet-50 loaded successfully.")


# ============================================
# Same preprocessing as A0
# ============================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================
# Prediction function
# ============================================

def predict(image):

    if image is None:
        return {}

    image = image.convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0]

    top_values, top_indices = torch.topk(
        probabilities,
        k=min(10, NUM_CLASSES)
    )

    results = {}

    for value, index in zip(
        top_values,
        top_indices
    ):
        disease = DISEASE_LABELS[index.item()]
        percentage = float(value.item() * 100)

        results[disease] = percentage

    return results


# ============================================
# Gradio interface
# ============================================

description = """
### Multi-Disease Retinal AI Research Prototype

Upload a retinal fundus image to obtain the model's
top predicted disease probabilities.

**Model:** ResNet-50  
**Dataset:** RFMiD  
**Task:** Multi-label retinal disease classification  
**Classes:** 45

⚠️ **Research/Educational Use Only**

This model is a research prototype and is NOT intended
for medical diagnosis, treatment, or clinical decision-making.
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload Retinal Fundus Image"
    ),
    outputs=gr.Label(
        num_top_classes=10,
        label="Top Predictions"
    ),
    title="Retinal AI — Multi-Disease Classification",
    description=description,
    flagging_mode="never"
)


# ============================================
# Launch
# ============================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )

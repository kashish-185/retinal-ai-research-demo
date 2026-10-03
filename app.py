import os

import gradio as gr
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image


# ============================================================
# Configuration
# ============================================================

HF_REPO_ID = "Kashish-goel185/retinal-ai-resnet50"

MODEL_FILE = "reproduced_A0_resnet50.onnx"
MODEL_DATA_FILE = "reproduced_A0_resnet50.onnx.data"

NUM_CLASSES = 45


DISEASE_LABELS = [
    "DR",
    "ARMD",
    "MH",
    "DN",
    "MYA",
    "BRVO",
    "TSLN",
    "ERM",
    "LS",
    "MS",
    "CSR",
    "ODC",
    "CRVO",
    "TV",
    "AH",
    "ODP",
    "ODE",
    "ST",
    "AION",
    "PT",
    "RT",
    "RS",
    "CRS",
    "EDN",
    "RPEC",
    "MHL",
    "RP",
    "CWS",
    "CB",
    "ODPM",
    "PRH",
    "MNF",
    "HR",
    "CRAO",
    "TD",
    "CME",
    "PTCR",
    "CF",
    "VH",
    "MCA",
    "VS",
    "BRAO",
    "PLQ",
    "HPED",
    "CL",
]


# ============================================================
# Download ONNX model from Hugging Face
# ============================================================

print("Downloading ONNX model from Hugging Face...")

MODEL_PATH = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=MODEL_FILE
)

MODEL_DATA_PATH = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=MODEL_DATA_FILE
)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}"
    )

if not os.path.exists(MODEL_DATA_PATH):
    raise FileNotFoundError(
        f"ONNX external data file not found: {MODEL_DATA_PATH}"
    )


print("ONNX model downloaded successfully.")
print("Model:", MODEL_PATH)
print("External data:", MODEL_DATA_PATH)


# ============================================================
# ONNX Runtime CPU Session
# ============================================================

print("Creating ONNX Runtime CPU session...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name

print("Input:", INPUT_NAME)
print("Output:", OUTPUT_NAME)

print("ONNX Runtime initialized successfully.")


# ============================================================
# Image Preprocessing
# Same preprocessing used by A0 ResNet-50
# ============================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def preprocess_image(image):
    """
    Convert uploaded retinal image into the exact
    input format expected by the trained A0 model.
    """

    image = image.convert("RGB")

    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image_array = (
        image_array - MEAN
    ) / STD

    # HWC -> CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array.astype(np.float32)


# ============================================================
# Numerically Stable Sigmoid
# ============================================================

def sigmoid(x):
    """
    Numerically stable sigmoid function.
    """

    result = np.empty_like(
        x,
        dtype=np.float32
    )

    positive = x >= 0

    result[positive] = (
        1.0 /
        (
            1.0 +
            np.exp(-x[positive])
        )
    )

    exp_x = np.exp(x[~positive])

    result[~positive] = (
        exp_x /
        (1.0 + exp_x)
    )

    return result


# ============================================================
# Prediction
# ============================================================

def predict(image):
    """
    Run multi-label retinal disease prediction.
    """

    if image is None:
        return {}

    try:

        input_tensor = preprocess_image(
            image
        )

        logits = session.run(
            [OUTPUT_NAME],
            {
                INPUT_NAME: input_tensor
            }
        )[0]

        probabilities = sigmoid(
            logits[0]
        )

        # Top 10 predictions
        top_indices = np.argsort(
            probabilities
        )[::-1][:10]

        predictions = {
            DISEASE_LABELS[int(index)]:
            float(probabilities[int(index)])
            for index in top_indices
        }

        return predictions

    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return {}


# ============================================================
# Medical Disclaimer
# ============================================================

DISCLAIMER = """
### Important

This is a **research demonstration** and is **not a medical
diagnostic tool**. Predictions should not be used for clinical
diagnosis or treatment decisions.

The model is provided for educational and research purposes only.
"""


# ============================================================
# Gradio Interface
# ============================================================

demo = gr.Interface(
    fn=predict,

    inputs=gr.Image(
        type="pil",
        label="Upload Retinal Fundus Image"
    ),

    outputs=gr.Label(
        num_top_classes=10,
        label="Predicted Disease Probabilities"
    ),

    title="Retinal AI Research Demonstration",

    description=(
        "Multi-disease retinal fundus image prediction "
        "using a ResNet-50 model converted to ONNX."
    ),

    article=DISCLAIMER,

    examples=None,

    flagging_mode="never"
)


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "10000"
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )

# Retinal AI Research Demo

A research and educational prototype for multi-disease
retinal fundus image classification.

## Model

- Architecture: ResNet-50
- Dataset: RFMiD
- Task: Multi-label retinal disease classification
- Number of disease classes: 45
- Input size: 224 x 224
- Activation: Sigmoid
- Framework: PyTorch
- Web interface: Gradio

## Deployment

The trained model checkpoint is stored separately in a
Hugging Face model repository.

The application downloads the model automatically when
the web application starts.

## Repository Structure

- `app.py` - Gradio web application
- `model_config.json` - Model configuration and disease labels
- `requirements.txt` - Python dependencies
- `.python-version` - Python version used for deployment
- `.gitignore` - Prevents model checkpoints from being committed

## Research Notice

This project is intended for research and educational
demonstration only.

It is not intended for medical diagnosis, treatment,
or clinical decision-making.

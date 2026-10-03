# RFMiD Multi-Disease Retinal AI

A research and educational prototype for **multi-label retinal disease classification** using a **ResNet-50** deep learning model trained on the **RFMiD (Retinal Fundus Multi-Disease Image Dataset)**.

The project provides a lightweight web interface for demonstrating automated retinal-image classification and model inference.

## Overview

This project explores automated multi-label classification of retinal fundus images, where a single image can receive predictions across multiple disease categories.

Given a retinal fundus photograph, the application:

- preprocesses the image to the model's expected input format
- runs inference with the trained ResNet-50 model
- applies a sigmoid activation to obtain per-class probabilities
- displays predictions across all 45 supported disease classes

> **Research status:** This project is intended for academic demonstration, experimentation, and research purposes. The outputs are model predictions and should not be interpreted as clinical findings.

## Model

| Component | Configuration |
|---|---|
| Architecture | ResNet-50 |
| Dataset | RFMiD |
| Task | Multi-label retinal disease classification |
| Disease classes | 45 |
| Input size | 224 × 224 |
| Output | Per-class probabilities |
| Activation | Sigmoid |
| Classification threshold | 0.5 |
| Framework | PyTorch |
| Normalization | ImageNet mean/std |

## Repository Structure

```text
retinal-ai-research-demo/
├── app.py
├── model_config.json
├── requirements.txt
├── .python-version
└── README.md

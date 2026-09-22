# ⚖️ NyayaLlama: Indian Legal AI Assistant

NyayaLlama is an AI-powered Indian legal information assistant built
using Llama 3.2 3B Instruct and LoRA fine-tuning.

## 📌 Project Overview

NyayaLlama adapts a pretrained Llama 3.2 3B language model for
Indian legal question answering using parameter-efficient LoRA
fine-tuning.

The system is designed to provide clear and educational explanations
of Indian legal concepts.

## 🏗️ Architecture

User Question
      ↓
Llama 3.2 3B Instruct
      +
LoRA Adapter
      ↓
NyayaLlama
      ↓
Indian Legal Information Response

## 🛠️ Technologies

- Python
- PyTorch
- Llama 3.2 3B Instruct
- LoRA
- PEFT
- Unsloth
- Transformers
- BitsAndBytes
- Gradio
- Hugging Face

## 📚 Dataset

The model was fine-tuned using Indian legal question-answer data.

## 🧠 Fine-Tuning

Parameter-efficient fine-tuning was performed using LoRA.

The trained adapter is available on Hugging Face:

https://huggingface.co/bhuvanteja03/nyayallama-final-adapter

## 🚀 Demo

The project includes an interactive Gradio prototype.

The public demo link is provided through the project landing page
when the Colab inference session is running.

## 📂 Repository Structure

```text
NyayaLlama-Indian-Legal-AI/
│
├── NyayaLlama_Indian_Legal_AI.ipynb
├── app.py
├── requirements.txt
└── README.md

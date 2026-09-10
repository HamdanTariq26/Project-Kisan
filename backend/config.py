import os
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig
from kaggle_secrets import UserSecretsClient


# Use 8B for detailed response
# The notebook ultimately sets Use_8B = False, so the default remains the fine-tuned 3B model.
# Use Fine Tuned for fast and point to point responses.
# Model selection
USE_8B = False

MODEL_8B_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_3B_ID = "meta-llama/Llama-3.2-3B-Instruct"

BASE_MODEL_ID = MODEL_8B_ID if USE_8B else MODEL_3B_ID

checkpoint_path = os.getenv(
    "CHECKPOINT_PATH",
    "/kaggle/input/notebooks/hamdantariq/assistant/llama-agriculture/checkpoint-3500",
)


LOCAL_FAISS_PATH = os.getenv(
    "LOCAL_FAISS_PATH",
    "/kaggle/input/datasets/rohaantariq/faiss-new/FAISS-NEW",
)
DIRECTORY_PATH = Path(
    os.getenv(
        "DIRECTORY_PATH",
        "/kaggle/input/datasets/hamdantariq/rag-books",
    )
)

bnb_Config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def get_huggingface_secret() -> str:
    secret_label = "Hugging Face Access"
    return UserSecretsClient().get_secret(secret_label)


def get_ngrok_token() -> str:
    return UserSecretsClient().get_secret("NGROK_AUTH_TOKEN")

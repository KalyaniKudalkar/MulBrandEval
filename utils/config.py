# utils/config.py
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
HF_TOKEN        = os.getenv("HF_TOKEN")
DEEPL_AUTH_KEY  = os.getenv("DEEPL_AUTH_KEY")
SD_MODEL_ID     = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
FLUX_MODEL_ID   = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev")
MCLIP_MODEL_ID  = os.getenv("MCLIP_MODEL_ID", "sentence-transformers/clip-ViT-B-32-multilingual-v1")
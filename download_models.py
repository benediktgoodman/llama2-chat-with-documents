"""
Author: Benedikt Goodman, Division for National Accounts, Statistics Norway
Date: 29/02/2024
"""

from pathlib import Path
import os
from transformers import AutoTokenizer, AutoModel
import torch.cuda
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    
    # Activate cuda if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    HF_CACHE = Path.cwd().joinpath('model_cache')

    if not HF_CACHE.exists():
        HF_CACHE.mkdir()

    # Make os path var as well because langchain cant handle Pathlib paths >:(
    HF_CACHE_W_PATH = os.getcwd() + "\model_cache"

    embedding_model = "all-mpnet-base-v2"
    inference_model = "bardsai/jaskier-7b-dpo-v5.6"

    def download_embeddings_model(model_name, cache_path):
        HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                cache_folder = cache_path
            )
        return

    def download_inference_model(base_model_name: str, cache_path: str):
        """Downloads chosen huggingface model to cache_dir"""
        AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_path)
        AutoModel.from_pretrained(base_model_name, cache_dir=cache_path)
        return
    
    download_embeddings_model(embedding_model, HF_CACHE_W_PATH)
    download_inference_model(inference_model, HF_CACHE_W_PATH)
    
if __name__ == "__main__":
    main()
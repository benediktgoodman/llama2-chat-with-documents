import os
# from typing import List
from pathlib import Path

import torch

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl

# Activate cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
HF_CACHE = Path.cwd().joinpath('model_cache')

if not HF_CACHE.exists():
    HF_CACHE.mkdir()
    
LLM_MODEL = "model_cachemodels--bardsai--jaskier-7b-dpo-v5.6"
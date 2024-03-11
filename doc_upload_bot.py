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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)

# Activate cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths for project
DB_DIR: str = Path.cwd().joinpath('vectorstore.db')
HF_CACHE = Path.cwd().joinpath('model_cache')

if not HF_CACHE.exists():
    HF_CACHE.mkdir()
    
# Make os path var as well because langchain cant handle Pathlib paths >:(
HF_CACHE_W_PATH = os.getcwd() + "\model_cache"
EMBEDDING_MODEL = "all-mpnet-base-v2"
INFERENCE_MODEL = "model_cachemodels--bardsai--jaskier-7b-dpo-v5.6"


prompt_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
The example of your response should be:

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt(prompt_template):
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt

def load_model(
    model_path="model_cache/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    model_type="mistral",
    max_new_tokens=983,
    temperature=0.7,
    gpu_layers=200,
):
    """
    Load a locally downloaded model.

    Parameters:
        model_path (str): The path to the model to be loaded.
        model_type (str): The type of the model.
        max_new_tokens (int): The maximum number of new tokens for the model.
        temperature (float): The temperature parameter for the model.

    Returns:
        CTransformers: The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        SomeOtherException: If the model file is corrupt.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    llm = CTransformers(
        model=model_path,
        model_type=model_type,
        max_new_tokens=max_new_tokens,  # type: ignore
        temperature=temperature,  # type: ignore
        gpu_layers=gpu_layers,
        config = {'context_length' : 2048}
    )

    return llm

def get_vectorstore(
    model_name=EMBEDDING_MODEL,
    model_cache=HF_CACHE_W_PATH,
    persist_dir="vectorstore.db",
    device="cuda",
):
    """
    This function creates a retrieval-based question-answering bot.

    Parameters:
        model_name (str): The name of the model to be used for embeddings.
        persist_dir (str): The directory to persist the database.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').

    Returns:
        RetrievalQA: The retrieval-based question-answering bot.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
        SomeOtherException: If there is an issue with loading the embeddings or the model.
    """

    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"No directory found at {persist_dir}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            cache_folder = HF_CACHE_W_PATH
        )
    except Exception as e:
        raise Exception(
            f"Failed to load embeddings with model name {model_name}: {str(e)}"
        )

    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    return db

def create_retrieval_qa_chain(llm, prompt, db, num_matches=3):
    """
    Creates a Retrieval Question-Answering (QA) chain using a given language model, prompt, and database.

    This function initializes a RetrievalQA object with a specific chain type and configurations,
    and returns this QA chain. The retriever is set up to return the top 3 results (k=3).

    Args:
        llm (any): The language model to be used in the RetrievalQA.
        prompt (str): The prompt to be used in the chain type.
        db (any): The database to be used as the retriever.

    Returns:
        RetrievalQA: The initialized QA chain.
    """
    # Gives the QAbot memory
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": num_matches}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        memory=memory,
    )
    return qa_chain

@cl.on_chat_start
async def on_chat_start():
    # files = None

    # # Wait for the user to upload a file
    # while files is None:
    #     files = await cl.AskFileMessage(
    #         content="Please upload a text file to begin!",
    #         accept=["text/plain"],
    #         max_size_mb=20,
    #         timeout=180,
    #     ).send()

    # file = files[0]

    # msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    # await msg.send()

    # with open(file.path, "r", encoding="utf-8") as f:
    #     text = f.read()

    # # Split the text into chunks
    # texts = text_splitter.split_text(text)

    # # Create a metadata for each chunk
    # metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # # Create a Chroma vector store
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=EMBEDDING_MODEL,
    #     model_kwargs={"device": device},
    #     cache_folder = HF_CACHE_W_PATH,
        
    #     )
    
    # docsearch = await cl.make_async(Chroma.from_texts)(
    #     texts, embeddings, metadatas=metadatas
    # )

    db = get_vectorstore()
    
    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    try:
        llm = load_model()
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")
    
    # Set instructions for model
    prompt = (
        set_custom_prompt(prompt_template)
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        #retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # # Let the user know that the system is ready
    # msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    # await msg.update()

    #print(qa_chain.keys())
    
    cl.user_session.set("chain", qa_chain)
    
# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
#     cb = cl.AsyncLangchainCallbackHandler()

#     res = await chain.acall(message.content, callbacks=[cb])
#     answer = res["answer"]
#     source_documents = res["source_documents"]  # type: List[Document]

#     text_elements = []  # type: List[cl.Text]

#     if source_documents:
#         for source_idx, source_doc in enumerate(source_documents):
#             source_name = f"source_{source_idx}"
#             # Create the text element referenced in the message
#             text_elements.append(
#                 cl.Text(content=source_doc.page_content, name=source_name)
#             )
#         source_names = [text_el.name for text_el in text_elements]

#         if source_names:
#             answer += f"\nSources: {', '.join(source_names)}"
#         else:
#             answer += "\nNo sources found"

#     await cl.Message(content=answer, elements=text_elements).send()
    

@cl.on_message
async def process_chat_message(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    response = await chain.acall(message.content, callbacks=[cb])
    
    bot_answer = response["result"]
    #bot_answer = response["output"]
    source_documents = response["source_documents"]
    
    text_elements = []
    
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            bot_answer += f"\nSources: {', '.join(source_names)}"
        else:
            bot_answer += "\nNo sources found"

    await cl.Message(content=bot_answer).send()
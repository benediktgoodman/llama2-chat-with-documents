import os

import chainlit as cl
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

from langchain.memory import ChatMessageHistory, ConversationBufferMemory

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


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt


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


def load_model(
    model_path="model_cache/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    model_type="mistral",
    max_new_tokens=512,
    temperature=0.7,
    gpu_layers=50,
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

    # Additional error handling could be added here for corrupt files, etc.

    llm = CTransformers(
        model=model_path,
        model_type=model_type,
        max_new_tokens=max_new_tokens,  # type: ignore
        temperature=temperature,  # type: ignore
        gpu_layers=gpu_layers
    )

    return llm


def create_retrieval_qa_bot(
    model_name="model_cache/sentence-transformers_all-mpnet-base-v2",
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
            model_name=model_name,
            model_kwargs={"device": device},
        )
    except Exception as e:
        raise Exception(
            f"Failed to load embeddings with model name {model_name}: {str(e)}"
        )

    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    try:
        llm = load_model()  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    qa_prompt = (
        set_custom_prompt()
    )  # Assuming this function exists and works as expected

    try:
        qa = create_retrieval_qa_chain(
            llm=llm, prompt=qa_prompt, db=db
        )  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa


def retrieve_bot_answer(query):
    """
    Retrieves the answer to a given query using a QA bot.

    This function creates an instance of a QA bot, passes the query to it,
    and returns the bot's response.

    Args:
        query (str): The question to be answered by the QA bot.

    Returns:
        dict: The QA bot's response, typically a dictionary with response details.
    """
    qa_bot_instance = create_retrieval_qa_bot()
    bot_response = qa_bot_instance({"query": query})
    return bot_response

@cl.on_chat_start
async def initialize_bot():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    
    qa_chain = create_retrieval_qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Llama2 and LangChain."
    )
    await welcome_message.update()
    
    cl.user_session.set("chain", qa_chain)


@cl.on_message
async def process_chat_message(message: cl.Message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    qa_chain = cl.user_session.get("chain")
    callback_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback_handler.answer_reached = True
    
    # Her feiler det...
    response = await qa_chain.acall(message, callbacks=[callback_handler])
    bot_answer = response["result"]
    source_documents = response["source_documents"]

    if source_documents:
        bot_answer += "\nSources:" + str(source_documents)
    else:
        bot_answer += "\nNo sources found"

    await cl.Message(content=bot_answer).send()

# import gradio as gr

# def load_markdown_splash_screen(markdown_file_path):
#     """
#     Load a Markdown file and return its content as a string.

#     Args:
#         markdown_file_path (str): The path to the Markdown file to be loaded.

#     Returns:
#         str: The content of the Markdown file.
#     """
#     with open(markdown_file_path, "r") as file:
#         markdown_content = file.read()
#     return markdown_content

# def display_splash_screen(markdown_content):
#     """
#     Display the splash screen content using Gradio's Markdown component.

#     Args:
#         markdown_content (str): The Markdown content to be displayed.

#     Returns:
#         gr.Markdown: Gradio Markdown component with the splash screen content.
#     """
#     return gr.Markdown(markdown_content)

# def chat_interface():
#     """
#     Create and run the Gradio interface for the chatbot.
#     """
#     # Load and display the splash screen
#     splash_screen_content = load_markdown_splash_screen("chainlit.md")
#     splash_screen = display_splash_screen(splash_screen_content)

#     # Define the Gradio interface components
#     with gr.Blocks() as demo:
#         splash_screen  # Display the splash screen at the top
#         with gr.Row():
#             question_input = gr.Textbox(label="Your question:")
#             submit_button = gr.Button("Submit")
#         response_output = gr.Textbox(label="Bot's response:", interactive=False)

#         # Define what happens when the submit button is clicked
#         def get_response(question):
#             bot_response = retrieve_bot_answer(question)  # Assuming this function returns a string
#             return bot_response

#         submit_button.click(fn=get_response, inputs=question_input, outputs=response_output)

#     demo.launch()

# # Uncomment this line to run the Gradio interface
# chat_interface(share=True)

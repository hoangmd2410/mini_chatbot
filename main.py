from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SemanticSplitterNodeParser
from tools import *
import config
from llama_index.core.agent.react.base import ReActAgent
import logging
from data_extraction import CustomVectorDatabase
import asyncio
from ultis import *
import gradio as gr
logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        handlers=[
            logging.StreamHandler()  # Send log messages to the terminal
        ]
    )

def initialize():
    logging.info("Initializing the chatbot...")
    try:
        Settings.llm = Ollama(config.OLLAMA['model_name'],
                            request_timeout=config.OLLAMA['request_timeout'],
                            context_window=config.OLLAMA['context_window'])
    except Exception as e:
        logging.error(f"Failed to initialize Ollama: {e}. Please check your Ollama configuration.")
    Settings.embed_model = HuggingFaceEmbedding(config.EMBEDDING_MODEL)
    Settings.splitter = SemanticSplitterNodeParser(buffer_size=config.SEMANTIC_SPLITTER_CONFIG['buffer_size'],
                                                    breakpoint_percentile_threshold=config.SEMANTIC_SPLITTER_CONFIG['breakpoint_percentile_threshold'],
                                                    embed_model=Settings.embed_model)
    overal_query_engine = asyncio.run(CustomVectorDatabase().create_query_engine())
    tools=[create_file_path_tool(),
        create_query_engine_tool(overal_query_engine),
        create_query_particular_file_tool()]
    Settings.agent = ReActAgent.from_tools(tools=tools,verbose=True)
    logging.info("Chatbot has been initialized successfully!")

def create_ui():
    with gr.Blocks() as chatbot_interface:
        #initialize config
        initialize()
        # Add box for OpenAI API key
        gr.Markdown("# Add your OpenAI API key below")
        with gr.Row():
            api_key_input = gr.Textbox(label="OpenAI API Key", placeholder="Enter your OpenAI API key here...")
        api_key_submit = gr.Button("Submit_Key",size="lg")
        notification = gr.Label(label="Notification", value="")
        api_key_submit.click(fn = update_llm, inputs=api_key_input, outputs=notification)
        
        gr.Markdown("# Simple Chatbot Interface")
        with gr.Row():
            query_input = gr.Textbox(label="Your Query", placeholder="Type your question here...")
            response_output = gr.Textbox(label="Chatbot Response", interactive=False)
        
        submit_button = gr.Button("Submit")
        submit_button.click(fn=chat_response, inputs=query_input, outputs=response_output)
    return chatbot_interface

if __name__ == "__main__":
    ui = create_ui()
    ui.launch()

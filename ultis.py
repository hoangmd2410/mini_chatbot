from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import os
import requests
from bs4 import BeautifulSoup
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from PyPDF2 import PdfReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging

def update_llm(openai_key:str):
    try:
        Settings.llm = OpenAI(model='gpt-4o',api_key=openai_key, max_tokens=2000)
        Settings.agent.agent_worker._llm = Settings.llm
        _ = Settings.llm.complete("Hello, World!")
        logging.info("OpenAI LLM has been updated successfully!")
        noti = "OpenAI LLM has been updated successfully!"
    except Exception as e:
        noti = f"Error: {e}"
    return noti

async def chat_response(input_text)->str:
    try:
        #clear the memory
        Settings.agent.chat_history.clear()
        response = await Settings.agent.achat(input_text)
        return response.response
    except Exception as e:
        return f"Error: {e}"

def process_pdf(file_path):
    # Read the PDF
    pdf_reader = PdfReader(file_path)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text()
    extra_info= {"file_path": file_path, "url":"", "islink": False}
    return Document(text=content, extra_info=extra_info)

def scrape_data(url="https://www.hkic.edu.hk/en/programmes/safety-training?active_cat_tab=safety-training"):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page {url}")  
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser') 
    # Extract all text from the webpage
    text = soup.get_text(separator='\n', strip=True)
    extra_info= {"file_path": "", "url":url, "islink": True}
    return  Document(text=text, extra_info=extra_info)
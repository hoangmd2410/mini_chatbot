from llama_index.core import Settings
from glob import glob
import config
from llama_index.core.tools import FunctionTool,QueryEngineTool,ToolMetadata,adapt_to_async_tool
from llama_index.core.base.base_query_engine import BaseQueryEngine
from ultis import *
import logging
async def find_exact_file_path(input:str)->str:
    """
    Finds the exact full file path in the 'data' folder based on the non-exact file path.
    Args:
    input: str: The non-exact file name or partial file name.
    """
    list_files = glob(f"{config.DEFAULT_LOCATION}/*.pdf")
    if not list_files:
        logging.warning(f"No files found in the data folder: {config.DEFAULT_LOCATION}")
        return None
    query = f"""Given list of available files: {list_files}, output the file path in the list that most similar to the string: {input}.ONLY OUTPUT the exact one file path which is most similar to the string, if no file path is similar, OUTPUT empty string. ONLY OUTPUT THE COMPLETE EXACT FILE PATH OR EMPTY STRING.
    """
    # query = ChatMessage(role="user", content=query)
    exact_file_name = await Settings.llm.acomplete(query)
    return exact_file_name.text

def create_file_path_tool(fn = find_exact_file_path, 
                          name="Find_Exact_file_path", 
                          description="Provide the exact file path in the 'data' folder from not-exact file name. Argument is input variable which is the non-exact file name or partial file name.The output is ONLY the exact complete file path if found, otherwise empty string.")->FunctionTool:
    return FunctionTool(async_fn=fn, metadata=ToolMetadata(name=name, description=description))


async def query_particular_file(file_path:str, prompt:str)->str:
    """
    Build a query engine for a particular file and answer the prompt.
    """
    vector_store = VectorStoreIndex.from_documents([process_pdf(file_path)])
    engine = vector_store.as_query_engine(similarity_top_k=config.QUERY_ENGINE_CONFIG['similarity_top_k'])
    response = await engine.aquery(prompt)
    return response.response

def create_query_particular_file_tool(fn = query_particular_file, 
                                      name="Query_Particular_File", 
                                      description="create a query engine for the particular file and answer the prompt. It receives 2 arguments file_path which is file path and prompt which is the question to be answered.")->FunctionTool:
    return adapt_to_async_tool(FunctionTool(async_fn=fn, 
                        metadata=ToolMetadata(name =name, description=description)))

def create_query_engine_tool(engine:BaseQueryEngine)->QueryEngineTool:
    return QueryEngineTool.from_defaults(query_engine=engine,
                                         name="Overall_Query_Engine", 
                                         description="Query Engine for overall search about question that not related to any specific file or document. Never use for summarizing or asking about a specific file or document.")
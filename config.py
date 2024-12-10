
SEMANTIC_SPLITTER_CONFIG = {
    'buffer_size': 1,
    'breakpoint_percentile_threshold':90,
}

EMBEDDING_MODEL = "all-MiniLM-L12-v2"

OLLAMA = {"model_name":"gemma2", 
          "request_timeout":60, 
          "context_window":40000}

DEFAULT_LOCATION ='data'
DEFAULT_URL_LIST = ['https://www.hkic.edu.hk/en/programmes/safety-training?active_cat_tab=safety-training']


QUERY_ENGINE_CONFIG = {'similarity_top_k':5}

VECTOR_STORE_CONFIG = {'host':'', 'port':None}
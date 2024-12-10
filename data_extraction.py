from typing import Any, List, Optional
from llama_index.core import VectorStoreIndex, Document,StorageContext
from llama_index.core.schema import TransformComponent
from PyPDF2 import PdfReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.vector_stores.types import BasePydanticVectorStore
import config
from ultis import *
from glob import glob
import asyncio

class CustomVectorDatabase:
    def __init__(self, 
                 name='mini-chatbot',
                 data_path=config.DEFAULT_LOCATION, 
                 urls=config.DEFAULT_URL_LIST,
                 host=config.VECTOR_STORE_CONFIG['host'],
                 port=config.VECTOR_STORE_CONFIG['port']):
        self.data_path = data_path
        self.urls = urls
        self.host = host
        self.port = port
        self.name = name
        self.transformations = [
            SemanticSplitterNodeParser(
                buffer_size=config.SEMANTIC_SPLITTER_CONFIG['buffer_size'],
                breakpoint_percentile_threshold=config.SEMANTIC_SPLITTER_CONFIG['breakpoint_percentile_threshold'],
                embed_model=HuggingFaceEmbedding()
            ),
            HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
        ]
        self.vector_store_index = None
        self.vector_store = self.create_vector_store()
        self.query_engine = self.create_query_engine()

    def create_vector_store(self) -> QdrantVectorStore:
        if self.host and self.port:
            client = QdrantClient(host=self.host, port=self.port)
            aclient = AsyncQdrantClient(host=self.host, port=self.port)
        else:
            client = QdrantClient(location=":memory:")
            aclient = AsyncQdrantClient(location=":memory:")
            logging.info("No host and port provided, using in-memory storage")
        vector_store = QdrantVectorStore(
            collection_name=self.name,
            client=client,
            aclient=aclient)
        return vector_store

    async def ingest_data(self) -> VectorStoreIndex:
        logging.info("Ingesting data...")
        pdf_documents = [process_pdf(pdf) for pdf in glob(f"{self.data_path}/*.pdf")]
        url_documents = [scrape_data(url) for url in self.urls]
        documents = pdf_documents + url_documents
        pipeline = IngestionPipeline(
            transformations=self.transformations,
            vector_store=self.vector_store
        )
        _= await pipeline.arun(documents=documents)
        vector_store_index = VectorStoreIndex.from_vector_store(self.vector_store,embed_model=Settings.embed_model)
        return vector_store_index

    async def create_query_engine(self):
        if not self.vector_store_index:
            self.vector_store_index = await self.ingest_data()
        return self.vector_store_index.as_query_engine(**config.QUERY_ENGINE_CONFIG)
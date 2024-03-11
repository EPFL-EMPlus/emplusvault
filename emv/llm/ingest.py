import random
import textwrap
from typing import List

import emv.utils
import emv.io.media
import emv.features.text
import emv.llm.retriever
import emv.llm.query
import emv.db.queries

from emv.settings import LLM_DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, LLM_ENDPOINT
from emv.db.dao import DataAccessObject
DataAccessObject().set_user_id(3)

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

LOG = emv.utils.get_logger()


def process_transcripts(feature_type: str = 'transcript+ner', page_size: int = 1000):
    total = emv.db.queries.count_features_by_type(feature_type, True)
    LOG.info(f'Total number of {feature_type} features: {total}')
    num_pages = (total // page_size) + 1
    for page in range(1, num_pages + 1):
        offset = (page - 1) * page_size
        docs = emv.db.queries.get_features_by_type_paginated(feature_type, page_size, offset, True)
        nodes = transform_transcripts_to_text_nodes(docs)
        yield nodes


def transform_transcripts_to_text_nodes(docs: List) -> List[TextNode]:
    nodes = []
    prev_node = None
    for doc in docs:
        transcript = doc['data']['transcript']
        feature_id = doc['feature_id']
        media_id = doc['media_id']
        node = TextNode(text=transcript, id_=f"{feature_id}")
        node.metadata = {
            'media_id': media_id,
            'feature_id': feature_id,
        }
        nodes.append(node)
        if prev_node and prev_node.metadata['media_id'] == node.metadata['media_id']:
            prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=node.id_
            )
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=prev_node.id_
            )
        prev_node = node
    return nodes


def get_indexer():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="camembert/camembert-large"
    )
    Settings.llm = Ollama(base_url=LLM_ENDPOINT, model="mistral", request_timeout=30.0)
    return Indexer()


class Indexer():
    def __init__(self):
        self.vector_store = PGVectorStore.from_params(
            database=LLM_DB_NAME,
            host=DB_HOST,
            password=DB_PASSWORD,
            port=DB_PORT,
            user=DB_USER,
            table_name="camembert-large",
            embed_dim=1024
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.query_engine = None

    def index_transcripts(self):
        for docs in process_transcripts():
            index = VectorStoreIndex.from_documents(
                docs, storage_context=self.storage_context, show_progress=True
            )        

    def load_index(self):
        index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        self.query_engine = index.as_query_engine()
        return self.query_engine
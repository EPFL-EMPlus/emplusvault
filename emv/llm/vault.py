import random
import textwrap
import orjson
from typing import List, Generator, Any, Optional 

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
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore


LOG = emv.utils.get_logger()


def transform_small_transcripts_to_text_nodes(docs: List) -> List[TextNode]:
    nodes = []
    prev_node = None
    for doc in docs:
        if not doc['embedding_size']:
            continue
        transcript = doc['data']['transcript']
        feature_id = doc['feature_id']
        media_id = doc['media_id']
        node = TextNode(text=transcript, id_=f"{feature_id}")
        node.metadata = {
            'media_id': media_id,
            'feature_id': feature_id,
        }
        node.embedding = orjson.loads(doc['embedding_1024'])
        # print(node.embedding, type(node.embedding), feature_id)
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


def process_small_transcripts(feature_type: str = 'transcript+ner', page_size: int = 1000) -> Generator[List[TextNode], None, None]:
    total = emv.db.queries.count_features_by_type(feature_type, True)
    LOG.info(f'Total number of {feature_type} features: {total}')
    num_pages = (total // page_size) + 1
    for page in range(1, num_pages + 1):
        offset = (page - 1) * page_size
        docs = emv.db.queries.get_features_by_type_paginated(feature_type, page_size, offset, True)
        nodes = transform_small_transcripts_to_text_nodes(docs)
        yield nodes


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

    def index_transcripts(self) -> None:
        for docs in process_small_transcripts():
            # if not embedding, add
            self.vector_store.add(docs)  

    # def load_index(self):
    #     index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
    #     self.query_engine = index.as_query_engine()
    #     return self.query_engine
            

class VaultRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 5,
    ) -> None:
        """Init params."""
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.query_mode = query_mode
        self.similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_str: str) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self.embed_model.get_query_embedding(query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.similarity_top_k,
            mode=self.query_mode,
        )
        query_result = self.vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores
    

class VaultLLM:
    def __init__(self):
        Settings.embed_model = HuggingFaceEmbedding(model_name="camembert/camembert-large")
        Settings.llm = Ollama(base_url=LLM_ENDPOINT, model="ollama/mixtral:8x7b-instruct-v0.1-fp16", request_timeout=30.0)
        self.indexer = Indexer()
        self.retriever = VaultRetriever(self.indexer.vector_store, Settings.embed_model)
        self.llm = Settings.llm
        # self.query_engine = self.indexer.load_index()

    def set_retriever_top_k(self, k: int) -> None:
        self.retriever.similarity_top_k = k

    def set_retrevier_query_mode(self, mode: str) -> None:
        self.retriever.query_mode = mode

    def index_transcripts(self) -> None:
        self.indexer.index_transcripts()

    def retrieve(self, query: str) -> List[NodeWithScore]:
        return self.retriever.retrieve(query)


def get_vault_llm_backend():
    return VaultLLM()
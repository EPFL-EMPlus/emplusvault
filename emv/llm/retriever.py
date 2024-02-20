from typing import List, Any

import emv.utils
import emv.features.text
import emv.db.queries


LOG = emv.utils.get_logger()


class DocumentRetriever:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentRetriever, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        LOG.debug('Initializing DocumentRetriever')
        self.embedder = emv.features.text.TextEmbedder()
    
    def fetch_similar(self, query: str, limit: int = 10, short_clips_only: bool = True) -> List[Any]:
        LOG.debug(f'Fetching similar documents for query: {query}')
        embeddings = self.embedder.encode(query)
        feats = emv.db.queries.get_topk_features_by_embedding(embeddings[0], limit, short_clips_only)
        return feats
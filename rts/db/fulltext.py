import pathlib
import meilisearch
import pandas as pd
import numpy as np
import requests
import time

import rts.utils


_client = None


def get_client(url: str = 'http://localhost:7700', master_key: str = '1234') -> meilisearch.Client:
    global _client
    if not _client:
        _client = meilisearch.Client(url, master_key)
    return _client


def get_or_create_media_index(client: meilisearch.Client) -> meilisearch.index.Index:
#   {
#   "displayedAttributes": [
#     "*"
#   ],
#   "searchableAttributes": [
#     "*"
#   ],
#   "filterableAttributes": [],
#   "sortableAttributes": [],
#   "rankingRules":
#   [
#     "words",
#     "typo",
#     "proximity",
#     "attribute",
#     "sort",
#     "exactness"
#   ],
#   "stopWords": [],
#   "synonyms": {},
#   "distinctAttribute": null,
#   "typoTolerance": {
#     "enabled": true,
#     "minWordSizeForTypos": {
#       "oneTypo": 5,
#       "twoTypos": 9
#     },
#     "disableOnWords": [],
#     "disableOnAttributes": []
#   },
#   "faceting": {
#     "maxValuesPerFacet": 100
#   },
#   "pagination": {
#     "maxTotalHits": 1000
#   }
# }
    
    try:
        m = client.get_index('media')
    except:
        client.create_index('media', {
            'primaryKey': 'guid'
        })
        
    time.sleep(1)
    m = client.get_index('media')
    
    here = pathlib.Path(__file__).parent.resolve()
    stop_words = rts.utils.obj_from_json(here / 'stop_words_french.json')

    m.update_settings({
        'searchableAttributes': [
            'mediaId',
            'categoryName',
            'assetType',
            'contentType',
            'backgroundType',
            'collection',
            'title',
            'resume',
            'published'
        ],
        'displayedAttributes': [
            'guid',
            'mediaId',
            'title',
            'resume',
            'categoryName',
            'assetType',
            'contentType',
            'backgoundType',
            'collection',
            'publishedDate'
        ],
        'filterableAttributes': [
            'categoryName',
            'assetType',
            'contentType',
            'backgoundType',
            'collection',
            'publishedDate'
        ],
        'sortableAttributes': [
            'published',
            'publishedDate'
        ],
        'stopWords': stop_words,
       }
    )
    return m


def index_media(df: pd.DataFrame, index: meilisearch.index.Index):
    df2 = df.drop(columns=['sequences'])
    for chunk in np.array_split(df2, 50):
        res = chunk.to_dict(orient='records')
        index.add_documents(res)
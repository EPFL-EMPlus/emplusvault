import requests
import time
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import numpy as np

def get_wikidata_id(entity, top_n=1, language='fr', sleep_time=0.1):
    if entity is None or entity == '':
        return None

    time.sleep(sleep_time)
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': language,
        'type': 'item',
        'limit': top_n,
        'search': entity
    }

    r = requests.get(url, params=params)
    data = r.json()
    return data['search']

def get_wikidata_label(qid):
    # Define the API endpoint URL and parameters
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbgetentities',
        'ids': qid,
        'props': 'labels',
        'languages': 'en',
        'format': 'json'
    }

    # Send the API request and parse the response
    response = requests.get(url, params=params).json()
    entities = response.get('entities', {})
    if qid in entities:
        return entities[qid]["labels"]
    else:
        return []

def get_property(qids, property_id):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Construct a comma-separated list of Qids for the SPARQL query
    qids_str = " ".join([f"wd:{qid}" for qid in qids])

    # Define the SPARQL query to retrieve the location property for the given Qids
    query = f"""SELECT ?entity ?property
                WHERE {{
                    VALUES ?entity {{ {qids_str} }}
                    ?entity wdt:{property_id} ?property .
                }}"""

    # Set the SPARQL query and return format
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the query and parse the results
    results = sparql.query().convert()

    # Extract the property from the query results
    properties = {}
    if results['results']['bindings']:
        for binding in results['results']['bindings']:
            qid = binding['entity']['value'].split('/')[-1]
            prop = binding['property']['value']
            #if prop.startswith("http://www.wikidata.org/entity/"):
            #    prop = get_wikidata_label(prop)
            properties[qid] = prop

    return properties

def process_batch(wiki_ids, property, BATCH_SIZE = 100):
    prop_results = {}
    for i in range(0, len(wiki_ids), BATCH_SIZE):
        time.sleep(1)
        batch = wiki_ids[i:i+BATCH_SIZE]
        try:
            batch_properties = get_property(batch, property)
            prop_results.update(batch_properties)
        except Exception as e:
            print(f"Error retrieving info for {batch}: {e}")
            continue

    return prop_results
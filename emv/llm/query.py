from typing import List, Any

import emv.utils
import emv.db.queries
import emv.settings

# import litellm
# import guidance
# from guidance import models, gen, select, user, assistant, system
import dspy

LOG = emv.utils.get_logger()


class QueryExpanderMoreContext(dspy.Signature):
    """Expand a query by generating more context and information about the original query in French"""
    query = dspy.InputField(desc="A query")
    expanded = dspy.OutputField(desc="expanded query with at least 500 tokens")


def expand_query(query: str) -> str:
    chain = dspy.ChainOfThought(QueryExpanderMoreContext)
    res = chain(query=query)
    return res.expanded

# class LLMWrapper:
#     _instance = None
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(LLMWrapper, cls).__new__(cls)
#             cls._instance._initialize()
#         return cls._instance

#     def _initialize(self):
#         LOG.info('Initializing LLMWrapper')
#         self.llm = models.LiteLLMCompletion(api_base=emv.settings.LLM_ENDPOINT, 
#                                             model='ollama/mixtral:8x7b-instruct-v0.1-fp16')
#         self.llm.litellm.set_verbose = False

#     def __getattr__(self, name):
#         """
#         Forward attribute access to the llm object.
#         """
#         # LOG.info(f'Forwarding {name} access to llm object')
#         attr = getattr(self.llm, name)

#         # If the attribute is callable (e.g., a method), return a wrapper function
#         # that will call the method on the llm object. Otherwise, return the attribute value.
#         if callable(attr):
#             def wrapper(*args, **kwargs):
#                 LOG.info(f'Calling method {name} with args={args}, kwargs={kwargs}')
#                 return attr(*args, **kwargs)
#             return wrapper
#         else:
#             return attr
        
#     def __add__(self, other):
#         # Forward the + operation to the self.llm object
#         return self.llm + other
    
#     def expand_query(self, query: str, num_tokens: int = 100) -> str:
#         LOG.info(f'Expanding query: {query}')
#         lm = None
#         with system():
#             lm = self.llm + '''\
#             You are an expert in human psychology. \ 
#             You are asked to expand the user query to include more relevant information or questions that would expand the text for more powerful text embeding and retrieval.
#             '''
#         with user():
#             lm += "Query: {query}"

#         with assistant():
#             lm += gen("expanded_query", max_tokens=num_tokens, stop='\n')
#         return lm
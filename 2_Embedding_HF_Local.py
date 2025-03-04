from langchain_huggingface import HuggingFaceEmbeddings

import os



# Set Hugging Face cache directory (optional)
os.environ['HF_HOME'] = 'D:/huggingface_cache_em'


Embeddings  = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

text  = 'Dehli is the Capital of India'

vector = Embeddings.embed_query(text)
print(str(vector))
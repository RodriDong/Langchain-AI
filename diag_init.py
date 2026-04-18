import sys
import traceback
from time import time

print('PID', end=': ')
print(sys.executable)
print('Start init', flush=True)
try:
    from langchain_community.embeddings import GPT4AllEmbeddings
    print('Imported GPT4AllEmbeddings', flush=True)
    t0 = time()
    emb = GPT4AllEmbeddings(model_file='models/all-MiniLM-L6-v2-f16.gguf')
    t1 = time()
    print('Initialized GPT4AllEmbeddings in', t1-t0, 'seconds', flush=True)
    print('Embedding object type:', type(emb), flush=True)
except Exception as e:
    print('Exception during init:', flush=True)
    traceback.print_exc()
    sys.exit(1)
print('Done', flush=True)


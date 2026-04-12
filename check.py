import os
print('vectorstore exists:', os.path.exists('vectorstore'))
print('books exists:', os.path.exists('books'))
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
e = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
db = Chroma(persist_directory='vectorstore', embedding_function=e)
ids = db.get()['ids']
print('chunks:', len(ids))
if ids:
    results = db.similarity_search('توحيد', k=2)
    print('نتائج البحث:', len(results))
    print(results[0].page_content[:100])
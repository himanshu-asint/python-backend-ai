import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load documents
docs_path = "docs"
all_docs = []
for file in os.listdir(docs_path):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(docs_path, file))
        all_docs.extend(loader.load())

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

# Embed and store with FAISS
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)
vectorstore.save_local("faiss_index")
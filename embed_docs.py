import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load and split all .txt files in docs/
docs = []

for file_name in os.listdir("./docs"):
    if file_name.endswith(".txt"):
        loader = TextLoader(f"./docs/{file_name}")
        text = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs += splitter.split_documents(text)

# Create embeddings using a specific model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS index
db = FAISS.from_documents(docs, embedding)
db.save_local("faiss_index")

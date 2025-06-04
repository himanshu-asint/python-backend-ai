# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama  # Local LLM

class Question(BaseModel):
    question: str

app = FastAPI()

# Load local embeddings and FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Local LLM via Ollama (adjust model name as per your setup)
llm = Ollama(model="mistral", num_predict=128)  # or use "llama2", "llama3", etc.

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

@app.post("/ask-ai")
async def ask_ai(question: Question):
    # Get docs and scores
    results = vectorstore.similarity_search_with_score(question.question, k=3)
    if not results:
        return {"answer": "Please ask questions related to the documents provided."}
    top_doc, score = results[0]
    source = top_doc.metadata.get('source', '')
    print(f"Top doc source: {source}, score: {score}")
    allowed_sources = [
        'materials_table.txt', 'fluid_data.txt', 'failure_library.txt',
        'api581.txt', 'api570.txt', 'api510.txt', 'greet.txt'
    ]
    if not any(source.endswith(s) for s in allowed_sources) or score > 1.5:
        return {"answer": "Please ask questions related to the documents provided."}
    # Otherwise, let the LLM answer
    result = qa_chain({"query": question.question})
    if not result["result"].strip():
        return {"answer": "I couldn't find a relevant answer in the documents."}
    return {
        "answer": result["result"],
        "sources": [doc.metadata["source"] for doc in result["source_documents"]]
    }

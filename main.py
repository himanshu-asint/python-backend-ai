from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

class Question(BaseModel):
    question: str

app = FastAPI()

# Load FAISS index and retriever
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Load Flan-T5 model
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

@app.post("/ask-ai")
async def ask_ai(q: Question):
    # Use similarity_search_with_score to get docs and scores
    results = vectorstore.similarity_search_with_score(q.question, k=10)
    if not results:
        return {"answer": "Please ask questions related to the documents provided."}
    top_doc, score = results[0]
    source = top_doc.metadata.get('source', '')
    print(f"Top doc source: {source}, score: {score}")
    # Always allow greeting if greet.txt is top doc
    if source.endswith('greet.txt'):
        response = qa_chain.invoke({"query": q.question})
        return {"answer": response['result']}
    # Only answer if top doc is from allowed sources and score is LOW enough (distance)
    allowed_sources = [
        'materials_table.txt', 'fluid_data.txt', 'failure_library.txt',
        'api581.txt', 'api570.txt', 'api510.txt'
    ]
    if any(source.endswith(s) for s in allowed_sources) and score <= 2.0:
        response = qa_chain.invoke({"query": q.question})
        return {"answer": response['result']}
    # Otherwise, fallback
    return {"answer": "Please ask questions related to the documents provided."}
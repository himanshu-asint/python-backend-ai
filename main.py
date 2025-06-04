from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

class Query(BaseModel):
    question: str

# Load a text-generation pipeline using a better model
model_name = "facebook/opt-350m"  # Using a more capable model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,  # Lower temperature for more focused answers
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Create a more focused prompt template
template = """You are a helpful AI assistant. Use the following context to answer the question. 
If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context: {context}

Question: {question}

Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

@app.post("/ask-ai")
async def ask_ai(query: Query):
    # Load the FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(
        "faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Create the QA chain with improved configuration
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_kwargs={
                "k": 5,  # Retrieve more relevant documents
                "score_threshold": 0.7  # Only use highly relevant documents
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    # Get the answer
    result = qa_chain.invoke({"query": query.question})
    return {"answer": result["result"]}

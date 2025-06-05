# main.py
from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama  # Local LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
from typing import List

class Question(BaseModel):
    question: str

app = FastAPI()

# Load local embeddings and FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if os.path.exists("faiss_index") and os.path.exists(os.path.join("faiss_index", "index.faiss")):
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
else:
    vectorstore = None  # No vectorstore yet

# Local LLM via Ollama (adjust model name as per your setup)
llm = Ollama(model="mistral", num_predict=128)  # or use "llama2", "llama3", etc.

if vectorstore is not None:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
else:
    retriever = None
    qa_chain = None

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/upload-and-train")
async def upload_and_train(files: List[UploadFile] = File(...)):
    global vectorstore, retriever, qa_chain
    try:
        # Process each uploaded file
        for file in files:
            # Save the file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load and process the text file
            loader = TextLoader(file_path)
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add chunks to vector store
            if vectorstore is None:
                vectorstore = FAISS.from_documents(chunks, embedding_model)
            else:
                vectorstore.add_documents(chunks)
        
        # Save the updated vector store
        if vectorstore is not None:
            vectorstore.save_local("faiss_index")
            # (Re)create retriever and qa_chain
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
        
        return {
            "status": "success",
            "message": f"Successfully processed {len(files)} files and updated the model",
            "files_processed": [file.filename for file in files]
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing files: {str(e)}"
        }

@app.post("/ask-ai")
async def ask_ai(question: Question):
    if vectorstore is None or qa_chain is None or len(vectorstore.docstore._dict) == 0:
        return {"answer": "No documents available. Please upload and train with new documents."}
    # Get docs and scores
    results = vectorstore.similarity_search_with_score(question.question, k=3)
   
    if (
        not results or
        len(results) == 0 or
        not isinstance(results[0], (list, tuple)) or
        len(results[0]) < 2
    ):
        return {"answer": "Please ask questions related to the documents provided."}
    top_doc, score = results[0]
    source = top_doc.metadata.get('source', '')
    print(f"Top doc source: {source}, score: {score}")
    if score > 1.5:
        return {"answer": "Please ask questions related to the documents provided."}
    # Otherwise, let the LLM answer
    result = qa_chain({"query": question.question})
    if not result["result"].strip():
        return {"answer": "I couldn't find a relevant answer in the documents."}
    return {
        "answer": result["result"]
    }

@app.post("/sync-uploads")
async def sync_uploads():
    global vectorstore, retriever, qa_chain
    try:
        existing_files = set(os.listdir(UPLOAD_DIR))
        if vectorstore is None:
            return {
                "status": "success",
                "removed_docs": 0,
                "message": "No vectorstore to sync."
            }
        all_docs = vectorstore.docstore._dict
        to_remove = []
        for doc_id, doc in all_docs.items():
            source = doc.metadata.get('source', '')
            filename = os.path.basename(source)
            if filename not in existing_files:
                to_remove.append(doc_id)
        for doc_id in to_remove:
            vectorstore.docstore._dict.pop(doc_id, None)
        remaining_docs = list(vectorstore.docstore._dict.values())
        if remaining_docs:
            new_vectorstore = FAISS.from_documents(remaining_docs, embedding_model)
            new_vectorstore.save_local("faiss_index")
            vectorstore = new_vectorstore
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
        else:
            import shutil
            try:
                shutil.rmtree("faiss_index")
            except Exception:
                pass
            vectorstore = None
            retriever = None
            qa_chain = None
        return {
            "status": "success",
            "removed_docs": len(to_remove),
            "message": f"Removed {len(to_remove)} documents whose source files are missing in uploads folder and rebuilt the index."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during sync: {str(e)}"
        }

@app.post("/delete-uploaded-doc")
async def delete_uploaded_doc(filename: str = Body(..., embed=True)):
    global vectorstore, retriever, qa_chain
    file_path = os.path.join(UPLOAD_DIR, filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            existing_files = set(os.listdir(UPLOAD_DIR))
            if vectorstore is None:
                return {
                    "status": "success",
                    "message": f"Deleted {filename} (file only, no vectorstore to sync)",
                    "removed_files": 1
                }
            all_docs = vectorstore.docstore._dict
            to_remove = []
            for doc_id, doc in all_docs.items():
                source = doc.metadata.get('source', '')
                fname = os.path.basename(source)
                if fname not in existing_files:
                    to_remove.append(doc_id)
            for doc_id in to_remove:
                vectorstore.docstore._dict.pop(doc_id, None)
            remaining_docs = list(vectorstore.docstore._dict.values())
            if remaining_docs:
                new_vectorstore = FAISS.from_documents(remaining_docs, embedding_model)
                new_vectorstore.save_local("faiss_index")
                vectorstore = new_vectorstore
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True
                )
            else:
                import shutil
                try:
                    shutil.rmtree("faiss_index")
                except Exception:
                    pass
                vectorstore = None
                retriever = None
                qa_chain = None
            return {
                "status": "success",
                "message": f"Deleted {filename} and synced vectorstore.",
                "removed_files": 1
            }
        else:
            return {
                "status": "error",
                "message": f"File {filename} does not exist in uploads.",
                "removed_files": 0
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error deleting file: {str(e)}",
            "removed_files": 0
        }

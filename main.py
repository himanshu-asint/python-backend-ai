# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import os
import base64
import numpy as np
import uuid
import torch
import torchvision.transforms as transforms # Added for image quality assessment
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from piq import brisque # Changed from BRISQUELoss to brisque for image quality assessment
from typing import Optional, List
from pydantic import BaseModel
from pypdf import PdfReader # Import PdfReader
import shutil # Import shutil for file operations
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader # Ensure PyMuPDFLoader is imported

# LangChain imports for LLM and document handling
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create UPLOAD_DIR first, then mount static files
UPLOAD_DIR = "uploads"
FAISS_INDEX_PATH = "faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# Initialize Grounding DINO model
print("Initializing Grounding DINO model...")
# You can choose a different Grounding DINO model if needed, e.g., "IDEA-Research/GroundingDINO-Tiny"
model_id = "IDEA-Research/grounding-dino-base"

# --- IMPORTANT: Replace <YOUR_HF_TOKEN> with your actual Hugging Face token ---
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN = "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"
# --------------------------------------------------------------------------------

try:
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, token=HF_TOKEN)
    model.eval() # Set model to evaluation mode
    print("Grounding DINO model loaded successfully.")
except Exception as e:
    print(f"Error loading Grounding DINO model: {e}")
    import traceback
    traceback.print_exc() 
    processor = None
    model = None

# Initialize BRISQUE model for image quality assessment
print("Initializing BRISQUE model...")
try:
    # BRISQUE is a function, not a class, so it doesn't need instantiation
    # We'll call it directly when needed in the /analyze-image endpoint.
    print("BRISQUE function ready for use.")
except Exception as e:
    print(f"Error setting up BRISQUE: {e}")

# Initialize Ollama LLM and Embeddings
print("Initializing Ollama LLM and Embeddings...")
llm = Ollama(model="mistral")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS vector store and QA chain initialization
vectorstore = None
qa_chain = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def initialize_vectorstore_and_qa_chain():
    global vectorstore, qa_chain
    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")) or not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.pkl")):
        print("FAISS index not found. Initializing empty vectorstore.")
        vectorstore = FAISS.from_documents([Document(page_content="", metadata={"source": ""})], embeddings)
        # Remove the dummy document if it's the only one
        if len(vectorstore.docstore._dict) == 1 and list(vectorstore.docstore._dict.values())[0].page_content == "":
            vectorstore.docstore._dict.clear()
            vectorstore.index_to_docstore_id.clear()
        vectorstore.save_local(FAISS_INDEX_PATH)
    else:
        print("Loading FAISS index...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    if len(vectorstore.docstore._dict) > 0:
        # Ensure the prompt template is general enough for various queries
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
        QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}), # Increased k to 10
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        print("QA chain initialized.")
    else:
        qa_chain = None
        print("No documents in vectorstore, QA chain not initialized.")

# Initialize at startup
initialize_vectorstore_and_qa_chain()

class ImageAnalysisResponse(BaseModel):
    damage_detected: bool
    damage_description: str
    damage_mask: Optional[str] = None
    image_quality_score: float
    image_quality_interpretation: str
    quality: str  # New field for good/bad quality classification
    highlighted_image_url: Optional[str] = None # Added for the URL of the processed image

class Question(BaseModel):
    question: str

def calculate_brisque_score(image):
    try:
        # Convert PIL Image to torch tensor
        if isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            img_np = np.array(image)
            # Convert to torch tensor and normalize to [0, 1]
            img_tensor = torch.from_numpy(img_np).float() / 255.0
            # Add batch dimension and convert to channels first format (B, C, H, W)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError("Input must be a PIL Image")

        # Calculate BRISQUE score
        score = brisque(img_tensor)
        return float(score.item())  # Convert tensor to float
    except Exception as e:
        print(f"Error calculating BRISQUE score: {e}")
        return 100.0  # Return a high score (bad quality) in case of error

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), request: Request = Request):
    """
    Analyze an image using Grounding DINO for zero-shot damage detection.
    Returns detected damage types with bounding boxes, a highlighted image URL, and an image quality score.
    """
    print(f"Received image analysis request for file: {file.filename}")
    
    if processor is None or model is None: # Check if Grounding DINO model loaded successfully
        raise HTTPException(status_code=503, detail="Grounding DINO model not loaded. Check server logs for details.")

    image_quality_score = None
    image_quality_interpretation = "Quality assessment not available."

    try:
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents)).convert("RGB") # Ensure RGB for Grounding DINO and BRISQUE
            print(f"Original image mode: {image.mode}, format: {image.format}")
        except Exception as e:
            print(f"Error opening or converting image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file or mode: {e}")

        # Calculate BRISQUE score if the function is available
        try:
            # Calculate BRISQUE score
            brisque_score = calculate_brisque_score(image)
            image_quality_score = round(brisque_score, 2)
            image_quality_interpretation = "Lower BRISQUE scores indicate better image quality."
            print(f"BRISQUE Score: {brisque_score}")
        except Exception as e:
            print(f"Error calculating BRISQUE score: {e}")
            image_quality_score = None
            image_quality_interpretation = "BRISQUE score could not be calculated."

        # Define the text query for damage types
        # Use ". " to separate different categories
        text_query = "burnt area . charred . melted . scorched . electrical damage . exposed wires . smoke damage . corrosion . crack . faulty component . circuit board . electronic component . power input . burned wires . damaged circuit board . heat damage . short circuit"
        print(f"Running Grounding DINO detection with query: '{text_query}'")

        # Process image and text prompt
        inputs = processor(images=image, text=text_query, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process outputs
        results = processor.post_process_grounded_object_detection(outputs, threshold=0.34, target_sizes=[image.size[::-1]])[0] # Increased threshold to 0.3
        print(f"Raw detection results: {results}") # Added for debugging
        
        detected_damage = []
        draw_image = image.copy() # Draw on the original image
        draw = ImageDraw.Draw(draw_image)
        
        # Try to load a truetype font for better text rendering
        try:
            font = ImageFont.truetype("arial.ttf", 30) # Increased font size for better visibility
        except IOError:
            font = ImageFont.load_default() # Fallback to default font

        padding = 5 # Define padding here

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # Filter for relevant damage types
            if any(damage_keyword in label.lower() for damage_keyword in [
                "burnt", "charred", "melted", "scorched", "electrical damage",
                "smoke damage", "damaged", "heat damage", "short circuit"
            ]):
                box = [round(i, 2) for i in box.tolist()]
                x1, y1, x2, y2 = box

                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Draw label and score
                label_text = f"{label}: {score:.2f}"
                if label_text.startswith("old pipe pipe"): # Check for the specific redundancy
                    label_text = label_text.replace("old pipe pipe", "old pipe", 1) # Replace only the first occurrence
                
                # Calculate text size to get its height and width for positioning
                temp_text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = temp_text_bbox[2] - temp_text_bbox[0]
                text_height = temp_text_bbox[3] - temp_text_bbox[1]

                # Adjust text position to be slightly inside the bounding box
                text_x = x1 + padding
                text_y = y2 - text_height - padding # Changed to place text at the bottom of the bounding box
                
                text_bbox = draw.textbbox((text_x, text_y), label_text, font=font)
                text_x1, text_y1, text_x2, text_y2_bbox = text_bbox # Use a different var name for y2 from text_bbox to avoid conflict
                
                # Add some padding around the text box (using already defined padding)
                padded_text_x1 = text_x1 - padding
                padded_text_y1 = text_y1 - padding
                padded_text_x2 = text_x2 + padding
                padded_text_y2 = text_y2_bbox + padding
                
                # Draw a semi-transparent background rectangle for the text
                # Using a light semi-transparent background to make text pop
                background_color = (0, 0, 0, 150) # Black with 150/255 alpha (semi-transparent)
                draw.rectangle([padded_text_x1, padded_text_y1, padded_text_x2, padded_text_y2], fill=background_color)
                
                # Draw the text itself
                draw.text((text_x, text_y), label_text, fill="white", font=font) # Changed text color to white for better contrast

                detected_damage.append({
                    "type": label,
                    "confidence": float(score),
                    "location": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })
        
        # Save the highlighted image
        highlighted_filename = f"detected_{uuid.uuid4().hex}_{file.filename}"
        highlighted_image_path = os.path.join(UPLOAD_DIR, highlighted_filename)
        draw_image.save(highlighted_image_path)
        
        # Construct the FULL URL for the highlighted image
        highlighted_image_url = str(request.base_url) + "static/" + highlighted_filename
        
        # Calculate image quality score using BRISQUE
        quality_score = calculate_brisque_score(image)
        
        # Determine quality category based on BRISQUE threshold
        quality = "good" if quality_score < 20 else "bad"
        
        return ImageAnalysisResponse(
            damage_detected=len(detected_damage) > 0,
            damage_description=", ".join([f"{damage['type']}: {damage['confidence']:.2f}" for damage in detected_damage]),
            damage_mask=None,
            image_quality_score=quality_score,
            image_quality_interpretation="Lower BRISQUE scores indicate better image quality.",
            quality=quality,
            highlighted_image_url=highlighted_image_url # Populate the new field
        )
        
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Vehicle Damage Detection API with Grounding DINO is running"}

# Pydantic model for ask-ai request
class AskAIRequest(BaseModel):
    query: str

# API endpoint to ask AI a question
@app.post("/ask-ai")
async def ask_ai(question: Question):
    if vectorstore is None or len(vectorstore.docstore._dict) == 0:
        return {"answer": "No documents available. Please upload and train with new documents."}

    # Greeting logic (if user's query is a simple greeting)
    greeting_keywords = ["hi", "hello", "hey", "greetings"]
    if any(keyword in question.question.lower() for keyword in greeting_keywords):
        return {"answer": "Hello! How can I help you today regarding the documents?"}
        
    # Get docs and scores
    results = vectorstore.similarity_search_with_score(question.question, k=3)
    
    # Debugging print statement for results
    print(f"Similarity search results: {results}")

    # Fallback message if no relevant documents are found above a certain threshold
    # Note: FAISS similarity scores are distances, so lower is better.
    # A score below a certain threshold (e.g., 1.5 or 2.0 based on observation) indicates relevance.
    if not results or len(results) == 0:
        return {"answer": "I can only answer questions related to the documents I have been trained on."}

    top_doc, score = results[0]
    source = top_doc.metadata.get('source', '')
    
    print(f"Top document score: {score}, source: {source}") # Debugging

    # Define allowed sources for in-context answers, including docs/greet.txt if applicable
    # This list will need to be dynamically managed or reflect actual uploaded docs.
    # For now, let's allow all documents if the score is good.
    
    # If the score is too high (i.e., not similar enough), provide fallback.
    # Adjusted threshold based on previous observation (1.18 for in-context, 1.78 for out-of-context)
    # A score <= 1.5 seems like a reasonable threshold for similarity.
    if score > 1.5: # Lower score means higher similarity
        return {"answer": "I can only answer questions related to the documents I have been trained on."}

    # Otherwise, let the LLM answer
    result = qa_chain({"query": question.question})
    if not result["result"].strip():
        return {"answer": "I couldn't find a relevant answer in the documents."}
    return {
        "answer": result["result"]
    }

# Pydantic model for upload-and-train request (single file)
class UploadFileResponse(BaseModel):
    filename: str
    message: str

@app.post("/upload-docs") # Changed route from upload-and-train to upload-docs
async def upload_docs(files: List[UploadFile] = File(...)):
    global vectorstore, qa_chain
    processed_files = []
    failed_files = {}

    try:
        # Process each uploaded file
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            # Save the file temporarily
            try:
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
            except Exception as e:
                failed_files[file.filename] = f"Failed to save file: {str(e)}"
                continue # Skip to the next file

            # Load and process the file based on extension
            documents = []
            try:
                if file_extension == ".txt":
                    loader = TextLoader(file_path)
                    documents = loader.load()
                elif file_extension == ".pdf":
                    loader = PyMuPDFLoader(file_path)
                    documents = loader.load()
                else:
                    failed_files[file.filename] = "Unsupported file type. Only .txt and .pdf are supported."
                    os.remove(file_path) # Clean up unsupported file
                    continue # Skip to the next file
            except Exception as e:
                 failed_files[file.filename] = f"Failed to load/process file: {str(e)}"
                 os.remove(file_path) # Clean up failed file
                 continue # Skip to the next file
            
            # Split text into chunks
            chunks = text_splitter.split_documents(documents)
            
            # Add chunks to vector store
            if vectorstore is None or len(vectorstore.docstore._dict) == 0:
                # Initialize vectorstore with the first batch of chunks
                if chunks:
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                else:
                    failed_files[file.filename] = "No content found in file after processing."
                    os.remove(file_path) # Clean up empty file
                    continue # Skip to the next file
            else:
                # Add to existing vectorstore
                if chunks:
                    vectorstore.add_documents(chunks)
                else:
                    failed_files[file.filename] = "No content found in file after processing."
                    os.remove(file_path) # Clean up empty file
                    continue # Skip to the next file

            processed_files.append(file.filename)

        # Save the updated vector store if it exists
        if vectorstore is not None:
            vectorstore.save_local(FAISS_INDEX_PATH)
            # (Re)create qa_chain if vectorstore was just created or updated
            prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
            QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_PROMPT}
            )

        return {
            "status": "success",
            "message": f"Successfully processed {len(processed_files)} files.",
            "files_processed": processed_files,
            "files_failed": failed_files
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"An unexpected error occurred during file processing: {str(e)}",
            "files_processed": processed_files,
            "files_failed": failed_files
        }

@app.post("/sync-uploads")
async def sync_uploads():
    global vectorstore, qa_chain
    try:
        existing_files = set(os.listdir(UPLOAD_DIR))
        
        # If vectorstore is None, but there are files in UPLOAD_DIR, we need to initialize it
        if vectorstore is None and existing_files:
            # Rebuild vectorstore from scratch if it was cleared
            print("Vectorstore is None but uploads exist. Rebuilding from scratch.")
            documents_to_process = []
            for filename in existing_files:
                file_path = os.path.join(UPLOAD_DIR, filename)
                file_extension = os.path.splitext(filename)[1].lower()
                try:
                    if file_extension == ".txt":
                        loader = TextLoader(file_path)
                        documents_to_process.extend(loader.load())
                    elif file_extension == ".pdf":
                        loader = PyMuPDFLoader(file_path)
                        documents_to_process.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {filename} during sync: {e}")
            
            if documents_to_process:
                chunks = text_splitter.split_documents(documents_to_process)
                if chunks:
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    vectorstore.save_local(FAISS_INDEX_PATH)
                    # Initialize QA chain as well
                    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
{context}
    
Question: {question}
Answer:"""
                    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": QA_PROMPT}
                    )
                    print(f"Rebuilt vectorstore from {len(documents_to_process)} documents.")
                    return {
                        "status": "success",
                        "removed_files": 0, # No files removed, rather added/synced
                        "message": "Vectorstore rebuilt from existing uploads."
                    }
                else:
                    return {
                        "status": "success",
                        "removed_files": 0,
                        "message": "No documents found in uploads folder to build vectorstore."
                    }
            else:
                return {
                    "status": "success",
                    "removed_files": 0,
                    "message": "No files in uploads folder to sync."
                }


        if vectorstore is None: # If still no vectorstore, means no files to sync
            return {
                "status": "success",
                "removed_files": 0,
                "message": "No vectorstore to sync as no files have been uploaded yet."
            }
        
        all_docs = vectorstore.docstore._dict
        docs_to_keep = []
        removed_file_count = 0
        
        # Identify documents whose original source files are missing
        for doc_id, doc in all_docs.items():
            source = doc.metadata.get('source', '')
            filename = os.path.basename(source) if source else None
            
            if filename and filename not in existing_files:
                removed_file_count += 1 # Count files, not chunks
            else:
                docs_to_keep.append(doc)

        if len(docs_to_keep) == len(all_docs):
            return {
                "status": "success",
                "removed_files": 0,
                "message": "No documents removed. All source files are present."
            }
        
        if docs_to_keep:
            # Rebuild the vectorstore from remaining documents
            new_vectorstore = FAISS.from_documents(docs_to_keep, embeddings)
            new_vectorstore.save_local(FAISS_INDEX_PATH)
            vectorstore = new_vectorstore
            # Recreate qa_chain
            prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
            QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_PROMPT}
            )
            message = f"Removed {removed_file_count} files whose source files are missing in uploads folder and rebuilt the index."
        else:
            # If no docs left, remove the index file if exists and set vectorstore/qa_chain to None
            try:
                shutil.rmtree(FAISS_INDEX_PATH)
            except Exception:
                pass
            vectorstore = None
            qa_chain = None
            message = "All documents removed. Vectorstore cleared."
            
        return {
            "status": "success",
            "removed_files": removed_file_count,
            "message": message
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error during sync: {str(e)}",
            "removed_files": 0
        }

@app.post("/delete-uploaded-doc")
async def delete_uploaded_doc(filename: str = Body(..., embed=True)):
    global vectorstore, qa_chain
    file_path = os.path.join(UPLOAD_DIR, filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            
            # Immediately trigger a sync and rebuild of the vectorstore
            existing_files = set(os.listdir(UPLOAD_DIR))
            
            if vectorstore is None or len(vectorstore.docstore._dict) == 0:
                 return {
                    "status": "success",
                    "message": f"Deleted {filename} (file only, no vectorstore to sync).",
                    "removed_files": 1
                }

            all_docs = vectorstore.docstore._dict
            docs_to_keep = []
            removed_chunks_count = 0 # Not directly needed for user, but useful for logic
            
            for doc_id, doc in all_docs.items():
                source = doc.metadata.get('source', '')
                fname = os.path.basename(source) if source else None
                
                if fname == filename:
                    removed_chunks_count += 1
                else:
                    docs_to_keep.append(doc)

            if docs_to_keep:
                new_vectorstore = FAISS.from_documents(docs_to_keep, embeddings)
                new_vectorstore.save_local(FAISS_INDEX_PATH)
                vectorstore = new_vectorstore
                
                # Recreate qa_chain
                prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
{context}
    
Question: {question}
Answer:"""
                QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": QA_PROMPT}
                )
                message = f"Deleted {filename} and synced vectorstore. Removed {removed_chunks_count} associated chunks." # Refined message
            else:
                try:
                    shutil.rmtree(FAISS_INDEX_PATH)
                except Exception:
                    pass
                vectorstore = None
                qa_chain = None
                message = f"Deleted {filename} and cleared vectorstore as no documents remain."
            
            return {
                "status": "success",
                "message": message,
                "removed_files": 1 # Always 1 file removed by this endpoint
            }
        else:
            return {
                "status": "error",
                "message": f"File {filename} does not exist in uploads.",
                "removed_files": 0
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error deleting file: {str(e)}",
            "removed_files": 0
        }

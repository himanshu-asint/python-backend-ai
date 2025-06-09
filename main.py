# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from io import BytesIO
import os
import base64
import numpy as np
import uuid
import random
import sys
import torch
from mobile_sam import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator

# Add the mobile_sam_repo directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mobile_sam_repo'))

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
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# Initialize MobileSAM
print("Initializing MobileSAM...")
sam_checkpoint = "mobile_sam.pt"

try:
    # Initialize the model using the proper MobileSAM architecture
    sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
    sam.to(device="cpu")
    
    # Initialize the mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    print("MobileSAM model loaded successfully.")
except FileNotFoundError:
    print(f"Error: MobileSAM checkpoint not found at {sam_checkpoint}. Please download it.")
    mask_generator = None
except Exception as e:
    print(f"Error loading MobileSAM model: {e}")
    import traceback
    traceback.print_exc()  # This will print the full error traceback
    mask_generator = None

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an image using MobileSAM for object segmentation.
    Returns detected masks as bounding boxes, area, IOU, and a highlighted image URL.
    """
    print(f"Received image analysis request for file: {file.filename}")
    
    if mask_generator is None: # Check if model loaded successfully
        raise HTTPException(status_code=503, detail="MobileSAM model not loaded. Check server logs for details.")

    try:
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents))
            print(f"Original image mode: {image.mode}, format: {image.format}")
            # Ensure image is in RGB mode for consistent processing by SAM
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"Image converted to RGB mode. New mode: {image.mode}")
            image_np = np.array(image)
        except Exception as e:
            print(f"Error opening or converting image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file or mode: {e}")
        
        # Ensure image_np is (H, W, 3) for SAM
        if image_np.ndim == 2: # Grayscale
            image_np = np.stack([image_np]*3, axis=-1)
        elif image_np.shape[2] == 4: # RGBA
            image_np = image_np[:, :, :3] # Take RGB channels only

        # Save original image temporarily
        original_temp_path = os.path.join(UPLOAD_DIR, f"original_{file.filename}")
        image.save(original_temp_path)

        # Generate masks using MobileSAM (everything mode)
        print("Running MobileSAM segmentation")
        masks = mask_generator.generate(image_np)
        print(f"Generated {len(masks)} masks.")
        
        # Prepare image for drawing masks
        draw_image = image.copy() # This is the RGB image
        # Convert draw_image to RGBA to properly composite transparent masks
        if draw_image.mode != 'RGBA':
            draw_image = draw_image.convert('RGBA')
        
        segmented_objects = []
        
        def get_random_color(alpha=100):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            return (r, g, b, alpha)

        for mask_data in masks:
            mask = mask_data['segmentation'] # This is a boolean numpy array (H, W)
            bbox = mask_data['bbox'] # [x, y, width, height]
            x, y, w, h = bbox
            
            # Create a colored overlay for the mask
            color_fill = get_random_color()
            colored_mask_np = np.zeros((*mask.shape, 4), dtype=np.uint8)
            colored_mask_np[mask] = list(color_fill)
            colored_mask_img = Image.fromarray(colored_mask_np)
            
            # Composite the colored mask onto the drawing image
            draw_image.alpha_composite(colored_mask_img)

            segmented_objects.append({
                "bbox": [x, y, w, h],
                "area": float(mask_data['area']),
                "predicted_iou": float(mask_data['predicted_iou']),
                "stability_score": float(mask_data['stability_score'])
            })

        # Save the masked image to the uploads directory
        highlighted_filename = f"masked_{uuid.uuid4().hex}_{file.filename}"
        masked_image_path = os.path.join(UPLOAD_DIR, highlighted_filename)
        draw_image.save(masked_image_path)
        
        # Construct the URL for the masked image
        masked_image_url = f"/static/{highlighted_filename}"
        
        # Clean up original temporary file
        try:
            os.remove(original_temp_path)
        except Exception as e:
            print(f"Error cleaning up original temporary file: {e}")

        return {
            "num_objects_found": len(segmented_objects),
            "segmented_objects": segmented_objects,
            "masked_image_url": masked_image_url,
            "message": "Image segmentation complete"
        }
        
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "SAM Image Segmentation API is running"}

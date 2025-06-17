from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
from predict import Predictor, model_cfg
from typing import List, Optional, Dict
from pydantic import BaseModel
import os
import json

app = FastAPI(
    title="SAN Semantic Segmentation API",
    description="API for Side Adapter Network (SAN) for Open-Vocabulary Semantic Segmentation",
    version="1.0.0"
)

# Initialize predictor with default model
predictor = Predictor(**model_cfg["san_vit_b_16"])

# Define output directory path
OUTPUT_DIR = "/san/output"

class SegmentationResponse(BaseModel):
    status: str
    segmentation: Dict[str, List]

@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(
    image: UploadFile = File(..., description="The input image file to segment"),
    vocabulary: str = Form(
        default="[]",
        description="JSON array of object names to segment. If empty, will use default vocabulary."
    ),
    augment_vocabulary: str = Form(
        default="COCO-all",
        description="Vocabulary expansion mode. Options: 'COCO-all' (all COCO categories) or 'COCO-stuff' (only stuff categories)"
    ),
    model_name: str = Form(
        default="san_vit_b_16",
        description="Model to use. Options: 'san_vit_b_16' or 'san_vit_large_16'"
    )
):
    """
    Segment an image using the SAN model.
    
    Args:
        image: The input image file to segment
        vocabulary: JSON array of object names to segment. If empty, will use default vocabulary.
        augment_vocabulary: Vocabulary expansion mode ("COCO-all" or "COCO-stuff")
        model_name: Model to use ("san_vit_b_16" or "san_vit_large_16")
    
    Returns:
        JSON response containing:
            - status: Success status
            - segmentation:
                - sem_seg: 2D list of segmentation indices
                - vocabulary: List of class names corresponding to the indices
    """
    try:
        print("received request")
        # Read and process image
        contents = await image.read()
        input_image = Image.open(io.BytesIO(contents))
        
        # Process vocabulary
        vocab_list = json.loads(vocabulary) if vocabulary else []
        print("vocab_list: ", vocab_list)
        
        # Get segmentation result
        result = predictor.predict(
            input_image,
            vocabulary=vocab_list,
            augment_vocabulary=augment_vocabulary
        )
        
        # Convert numpy array to list for JSON serialization
        sem_seg_list = result["sem_seg"].tolist()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success", 
                "segmentation": {
                    "sem_seg": sem_seg_list,
                    "vocabulary": result["vocabulary"]
                }
            }
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/models")
async def list_models():
    """List available models and their configurations"""
    return {"models": list(model_cfg.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860) 
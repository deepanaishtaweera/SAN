from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
from predict import Predictor, model_cfg
from typing import List, Optional
import base64

app = FastAPI(
    title="SAN Semantic Segmentation API",
    description="API for Side Adapter Network (SAN) for Open-Vocabulary Semantic Segmentation",
    version="1.0.0"
)

# Initialize predictor with default model
predictor = Predictor(**model_cfg["san_vit_b_16"])

@app.post("/segment")
async def segment_image(
    image: UploadFile = File(...),
    vocabulary: str = Form(""),
    augment_vocabulary: str = Form("COCO-all"),
    model_name: str = Form("san_vit_b_16"),
    visualization_mode: str = Form("overlay")
):
    """
    Segment an image using the SAN model.
    
    Args:
        image: The input image file
        vocabulary: Comma-separated list of object names to segment
        augment_vocabulary: Vocabulary expansion mode ("COCO-all" or "COCO-stuff")
        model_name: Model to use ("san_vit_b_16" or "san_vit_large_16")
        visualization_mode: Visualization mode ("overlay" or "mask")
    
    Returns:
        JSON response containing the segmentation result and visualization
    """
    try:
        # Read and process image
        contents = await image.read()
        input_image = Image.open(io.BytesIO(contents))
        
        # Process vocabulary
        vocab_list = [v.strip() for v in vocabulary.split(",")] if vocabulary else []
        
        # Get segmentation result
        result = predictor.predict(
            input_image,
            vocabulary=vocab_list,
            augment_vocabulary=augment_vocabulary
        )
        
        # Generate visualization
        vis_image = predictor.visualize(
            image=input_image,
            sem_seg=result["sem_seg"],
            vocabulary=result["vocabulary"],
            mode=visualization_mode
        )
        
        # Convert visualization to base64
        buffered = io.BytesIO()
        vis_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "status": "success",
            "segmentation": {
                "vocabulary": result["vocabulary"],
                "visualization": f"data:image/png;base64,{img_str}"
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/models")
async def list_models():
    """List available models"""
    return {"models": list(model_cfg.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860) 
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
from PIL import Image
import io
import numpy as np
from lang_sam import LangSAM
import supervision as sv
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import cv2

app = FastAPI()

# Enable CORS for all origins (Adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (Change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the langSAM model
langsam_model = LangSAM()

# Load SAM2 Model
sam2_checkpoint = "sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
device = torch.device("cpu")

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

def apply_mask(image, mask):
    """Overlay mask on image."""
    mask = mask.astype(np.uint8) * 255  # Convert mask to 0-255 scale
    mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_colored[mask > 0] = [30, 144, 255]  # Blue color for the mask
    
    # Add contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_colored, contours, -1, (255, 255, 255), thickness=2)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    return overlay


def draw_image(image_rgb, masks, xyxy, probs, labels):
    mask_annotator = sv.MaskAnnotator()
    # Create class_id for each unique label
    unique_labels = list(set(labels))
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_id = [class_id_map[label] for label in labels]

    # Add class_id to the Detections object
    detections = sv.Detections(
        xyxy=xyxy,
        mask=masks.astype(bool),
        confidence=probs,
        class_id=np.array(class_id),
    )
    annotated_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)
    return annotated_image


@app.post("/segment/sam2")
async def segment_image(
    file: UploadFile = File(...), 
    x: int = Form(...), 
    y: int = Form(...)
):
    """Segment image using SAM2 with a single input point."""
    image_bytes = await file.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image_pil)
    
    predictor.set_image(image_array)
    
    input_point = np.array([[x, y]])
    input_label = np.array([1])  # Foreground point
    
    # Run SAM2 model
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Get top mask
    top_mask = masks[np.argmax(scores)]

    # Apply mask overlay
    output_image = apply_mask(image_array, top_mask)

    # Convert to PNG
    output_pil = Image.fromarray(output_image)
    img_io = io.BytesIO()
    output_pil.save(img_io, format="PNG")
    img_io.seek(0)

    return Response(content=img_io.getvalue(), media_type="image/png")


@app.post("/segment/langsam")
async def segment_image(file: UploadFile = File(...), text_prompt: str = Form(...)):
    image_bytes = await file.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Run segmentation
    results = langsam_model.predict([image_pil], [text_prompt])
    
    # Convert to NumPy array
    image_array = np.asarray(image_pil)
    output_image = draw_image(
        image_array,
        results[0]["masks"],
        results[0]["boxes"],
        results[0]["scores"],
        results[0]["labels"],
    )
    
    # Convert back to PIL Image
    output_pil = Image.fromarray(np.uint8(output_image)).convert("RGB")
    
    # Save to byte stream
    img_io = io.BytesIO()
    output_pil.save(img_io, format="PNG")
    img_io.seek(0)
    
    return Response(content=img_io.getvalue(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

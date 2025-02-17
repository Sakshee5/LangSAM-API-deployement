from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
from PIL import Image
import io
import numpy as np
from lang_sam import LangSAM
import supervision as sv

app = FastAPI()

# Enable CORS for all origins (Adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (Change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the segmentation model
model = LangSAM()

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

@app.post("/segment/")
async def segment_image(file: UploadFile = File(...), text_prompt: str = Form(...)):
    image_bytes = await file.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Run segmentation
    results = model.predict([image_pil], [text_prompt])
    
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

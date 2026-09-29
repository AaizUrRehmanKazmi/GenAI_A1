from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from .preprocessing import validate_image

app = FastAPI(title="GenAI Assignment Skeleton", version="0.1.0")

@app.get("/health")
async def health():
    return {"status": "ok", "phase": "skeleton", "inference_ready": False}

async def pending(file: UploadFile):
    await validate_image(file)
    raise HTTPException(503, detail="Inference is not implemented. Complete the research, training and ONNX integration steps first.")

@app.post("/universal-restoration")
async def universal(file: UploadFile = File(...)):
    return await pending(file)

@app.post("/hard-routing")
async def hard(file: UploadFile = File(...)):
    return await pending(file)

@app.post("/soft-mixture")
async def soft(file: UploadFile = File(...)):
    return await pending(file)

@app.post("/face-to-sketch")
async def sketch(file: UploadFile = File(...), style: int = Form(..., ge=1, le=3)):
    return await pending(file)

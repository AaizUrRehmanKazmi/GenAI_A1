from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from .preprocessing import validate_image
from .task2_runtime import HardRoutingService
from .task3_runtime import SoftMixtureService
from .schemas import InferenceResult
from contextlib import asynccontextmanager
import logging
import os
from starlette.concurrency import run_in_threadpool

@asynccontextmanager
async def lifespan(app):
    app.state.hard_router = None
    try:
        app.state.hard_router = HardRoutingService(os.getenv('MODEL_DIR', '/models'))
    except Exception:
        logging.exception('Task 2 model initialization failed')
    app.state.soft_mixture = None
    try:
        app.state.soft_mixture = SoftMixtureService(os.getenv('TASK3_MODEL_DIR', '/models-task3'))
    except Exception:
        logging.exception('Task 3 model initialization failed')
    yield


app = FastAPI(title="GenAI Assignment", version="0.2.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "inference_ready": app.state.hard_router is not None,
            "tasks": {"hard-routing": app.state.hard_router is not None,
                      "soft-mixture": app.state.soft_mixture is not None}}

async def pending(file: UploadFile):
    await validate_image(file)
    raise HTTPException(503, detail="Inference is not implemented. Complete the research, training and ONNX integration steps first.")

@app.post("/universal-restoration")
async def universal(file: UploadFile = File(...)):
    return await pending(file)

@app.post("/hard-routing", response_model=InferenceResult)
async def hard(file: UploadFile = File(...)):
    payload = await validate_image(file)
    if app.state.hard_router is None:
        raise HTTPException(503, 'Task 2 models are unavailable. Check the mounted model bundle.')
    return await run_in_threadpool(app.state.hard_router.predict, payload)

@app.post("/soft-mixture", response_model=InferenceResult)
async def soft(file: UploadFile = File(...)):
    payload = await validate_image(file)
    if app.state.soft_mixture is None:
        raise HTTPException(503, 'Task 3 models are unavailable. Check the model bundle.')
    return await run_in_threadpool(app.state.soft_mixture.predict, payload)

@app.post("/face-to-sketch")
async def sketch(file: UploadFile = File(...), style: int = Form(..., ge=1, le=3)):
    return await pending(file)

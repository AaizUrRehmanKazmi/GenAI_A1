from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from .preprocessing import validate_image
from .task1_runtime import UniversalRestorationService
from .task2_runtime import HardRoutingService
from .task3_runtime import SoftMixtureService
from .task4_runtime import SketchService
from .schemas import InferenceResult
from contextlib import asynccontextmanager
import logging
import os
from starlette.concurrency import run_in_threadpool

@asynccontextmanager
async def lifespan(app):
    app.state.universal = None
    try:
        app.state.universal = UniversalRestorationService(os.getenv('TASK1_MODEL_DIR', '/models-task1'))
    except Exception:
        logging.exception('Task 1 model initialization failed')
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
    app.state.sketch = None
    try:
        app.state.sketch = SketchService(os.getenv("TASK4_MODEL_DIR", "/models-task4"))
    except Exception:
        logging.exception("Task 4 model initialization failed")
    yield


app = FastAPI(title="GenAI Assignment", version="0.3.0", lifespan=lifespan)

@app.get("/health")
async def health():
    universal_service = getattr(app.state, 'universal', None)
    hard_router = getattr(app.state, 'hard_router', None)
    soft_mixture = getattr(app.state, 'soft_mixture', None)
    sketch_service = getattr(app.state, 'sketch', None)
    return {"status": "ok",
            "inference_ready": any([universal_service, hard_router,
                                    soft_mixture, sketch_service]),
            "tasks": {"universal-restoration": universal_service is not None,
                      "hard-routing": hard_router is not None,
                      "soft-mixture": soft_mixture is not None,
                      "face-to-sketch": sketch_service is not None}}

@app.post("/universal-restoration")
async def universal(file: UploadFile = File(...)):
    payload = await validate_image(file)
    universal_service = getattr(app.state, 'universal', None)
    if universal_service is None:
        raise HTTPException(503, 'Task 1 model is unavailable. Check the mounted model bundle.')
    return await run_in_threadpool(universal_service.predict, payload)

@app.post("/hard-routing", response_model=InferenceResult)
async def hard(file: UploadFile = File(...)):
    payload = await validate_image(file)
    hard_router = getattr(app.state, 'hard_router', None)
    if hard_router is None:
        raise HTTPException(503, 'Task 2 models are unavailable. Check the mounted model bundle.')
    return await run_in_threadpool(hard_router.predict, payload)

@app.post("/soft-mixture", response_model=InferenceResult)
async def soft(file: UploadFile = File(...)):
    payload = await validate_image(file)
    soft_mixture = getattr(app.state, 'soft_mixture', None)
    if soft_mixture is None:
        raise HTTPException(503, 'Task 3 models are unavailable. Check the model bundle.')
    return await run_in_threadpool(soft_mixture.predict, payload)

@app.post("/face-to-sketch")
async def sketch(file: UploadFile = File(...), style: int = Form(..., ge=1, le=3)):
    payload = await validate_image(file)
    sketch_service = getattr(app.state, 'sketch', None)
    if sketch_service is None:
        raise HTTPException(503, 'Task 4 model unavailable. Check model bundle.')
    return await run_in_threadpool(sketch_service.predict, payload, style)

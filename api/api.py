"""
FastAPI Application for Text Generation

This API provides endpoints to generate text using your trained transformer model.

Endpoints:
- GET  /              → Health check
- POST /generate      → Generate text with custom parameters
- GET  /models/info   → Get model information
- POST /compare       → Compare different generation strategies

Run with:
    uvicorn api.api:app --reload --host 0.0.0.0 --port 8000

Then visit:
    http://localhost:8000/docs → Interactive API documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import yaml
import torch

from src.inference.generation import TextGeneration
from src.constants import CONFIG_PATH, BEST_MODEL

# Importing the schemas
from api.schema.schema import GenerateRequest, GenerateResponse, ModelInfo, CompareRequest, CompareResponse

# ------------------Fastapi Setup------------------------------------
app = FastAPI(
    title="Text Generation API",
    description="Text Generation using Decoder only Model",
    version="1.0.0",
    docs_url='/docs',
    redoc_url='/redocs'
)

# Add CORS middleware (allows frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------Global Variables--------------------------------------
# These will be initialized when the app starts
generator: Optional[TextGeneration] = None
model_info: Optional[Dict] = None

# -----------------Startup---------------------
@app.on_event('startup')
async def load_model():
    """
    This runs ONCE when the FastAPI app starts.
    We load the model here so we don't reload it for every request.
    """
    global generator, model_info

    print('-'*20)
    print("Loading model on startup")
    print('-'*20)

    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        device = config['device'] if torch.cuda.is_available() else "cpu"

        generator = TextGeneration(BEST_MODEL, CONFIG_PATH, device)
        if generator.model:
            print('Model Available')

        # Store model info
        model_info = {
            "model_name": "Decoder-Only Transformer",
            "vocabulary_size": generator.vocab_size,
            "total_parameters": sum(p.numel() for p in generator.model.parameters()),
            "max_sequence_length": config['data']['seq_length'],
            "device": str(device),
            "checkpoint_loaded": BEST_MODEL,
            "epoch": generator.checkpoint.get('epoch', 'unknown'),
            "step": generator.checkpoint.get('step', 'unknown'),
        }

        print(f"✓ Model loaded successfully!")
        print(f"  Parameters: {model_info['total_parameters']:,}")
        print(f"  Device: {model_info['device']}")
        print(f"  Checkpoint: {BEST_MODEL}")
        print(f"  Epoch: {model_info['epoch']}")
    
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        print("API will not work until model is loaded successfully!")
        raise


# --------------------API Endpoints---------------------------------
@app.get('/')
def root():
    """
    Health check endpoint.
    Returns basic info about the API.
    """
    if generator is None:
        return {
            "status": "error",
            "message": "Model not loaded. Check startup logs."
        }
    
    return {
        "status": "ready",
        "message": "Text Generation API is running!",
        "endpoints": {
            "generate": "/generate (POST)",
            "compare": "/compare (POST)",
            "model_info": "/models/info (GET)",
            "docs": "/docs (GET)"
        }
    }

@app.get('/health')
def get_health():
    """
    Detailed health check.
    """
    if generator is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "status": "ready",
        "model_loaded": True,
        "device": model_info['device']
    }

@app.get('/models/info', response_model=ModelInfo)
def get_model_info():
    """
    Get information about the loaded model.
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(**model_info)

@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    """
    Generate text from a prompt.
    
    This is the main endpoint for text generation.
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may still be starting up."
        )
    
    try:
        original_tokens = len(generator.tokenizer.encode(request.prompt))

        generated_response = generator.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            strategy=request.strategy,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty
        )

        # Count tokens in generated text
        total_tokens = len(generator.tokenizer.encode(generated_response))
        new_tokens = total_tokens - original_tokens

        return GenerateResponse(
            generated_text=generated_response,
            tokens_generated=new_tokens,
            strategy_used=request.strategy
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )
    
@app.post("/compare", response_model=CompareResponse)
def compare_strategies(request: CompareRequest):
    """
    Compare all generation strategies side-by-side.
    
    Useful for experimenting with different approaches.
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = generator.compare_strategies(
            prompt=request.prompt,
            max_length=request.max_length
        )
        
        return CompareResponse(
            prompt=request.prompt,
            results=results
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )
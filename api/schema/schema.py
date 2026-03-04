from pydantic import BaseModel, Field
from typing import Dict

class GenerateRequest(BaseModel):
    """
    Request body for text generation.
    
    These fields define what the user can send to the API.
    Pydantic validates types and sets defaults automatically.
    """
    prompt: str = Field(
        ...,  # ... means required
        description="Input text to continue generating from",
        example="The future of artificial intelligence is"
    )
    max_length: int = Field(
        50,
        ge=1,  # Greater than or equal to 1
        le=200,  # Less than or equal to 200
        description="Maximum number of tokens to generate",
    )
    strategy: str = Field(
        "combined",
        description="Generation strategy",
        pattern="^(greedy|temperature|top_k|top_p|combined)$"  # Only allow these values
    )
    temperature: float = Field(
        0.8,
        gt=0.0,  # Greater than 0
        le=2.0,  # Less than or equal to 2
        description="Sampling temperature (higher = more random)"
    )
    top_k: int = Field(
        50,
        ge=1,
        le=100,
        description="Number of top tokens to keep (for top_k strategy)"
    )
    top_p: float = Field(
        0.9,
        gt=0.0,
        le=1.0,
        description="Cumulative probability threshold (for top_p/combined strategy)"
    )
    repetition_penalty: float = Field(
        1.2,
        ge=1.0,
        le=2.0,
        description="Penalty for repeated tokens (1.0 = no penalty)"
    )

    class Config:
        # This makes the docs show example values
        json_schema_extra = {
            "example": {
                "prompt": "The history of artificial intelligence began",
                "max_length": 50,
                "strategy": "combined",
                "temperature": 0.8,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            }
        }


class GenerateResponse(BaseModel):
    """
    Response body for text generation.
    
    This defines the structure of what the API returns.
    """
    generated_text: str = Field(
        ...,
        description="The generated text (includes original prompt)"
    )
    tokens_generated: int = Field(
        ...,
        description="Number of new tokens generated"
    )
    strategy_used: str = Field(
        ...,
        description="The generation strategy that was used"
    )


class ModelInfo(BaseModel):
    """Information about the loaded model"""
    model_name: str
    vocabulary_size: int
    total_parameters: int
    max_sequence_length: int
    device: str
    checkpoint_loaded: str

class CompareRequest(BaseModel):
    """Request for comparing multiple strategies"""
    prompt: str = Field(..., description="Input prompt")
    max_length: int = Field(30, ge=1, le=100, description="Max tokens")


class CompareResponse(BaseModel):
    """Response with outputs from all strategies"""
    prompt: str
    results: Dict[str, str]  # strategy_name → generated_text

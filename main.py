# ============================================================
# app/main.py
# PURPOSE: FastAPI backend serving the full pipeline
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import time
import logging

# Import our pipeline components
from intent_classifier.model import IntentClassifier
from pipeline.document_processor import RetrievalPipeline
from synthesis.answer_engine import AnswerSynthesizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openlens")

app = FastAPI(
    title="OpenLens API",
    description="ML-Powered Research & Answer Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODELS ON STARTUP
# ============================================================
class AppState:
    classifier: Optional[IntentClassifier] = None
    retrieval_pipeline: Optional[RetrievalPipeline] = None
    synthesizer: Optional[AnswerSynthesizer] = None
    is_ready: bool = False

state = AppState()


@app.on_event("startup")
async def load_models():
    """Load all ML models when the server starts."""
    logger.info("Loading intent classifier...")
    state.classifier = IntentClassifier.load_model('models/intent_classifier.joblib')
    
    logger.info("Initializing retrieval pipeline...")
    state.retrieval_pipeline = RetrievalPipeline()
    
    logger.info("Initializing answer synthesizer...")
    state.synthesizer = AnswerSynthesizer()
    
    state.is_ready = True
    logger.info("✅ All models loaded. Server ready.")


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    mode: Optional[str] = Field(None, description="Force a specific mode")
    top_k: int = Field(5, ge=1, le=20)
    include_debug: bool = Field(False)


class SourceInfo(BaseModel):
    index: int
    title: str
    url: str
    source_type: str
    relevance_score: float


class QueryResponse(BaseModel):
    query: str
    detected_mode: str
    mode_confidence: float
    mode_was_forced: bool
    answer: str
    confidence: float
    key_points: List[str]
    sources: List[SourceInfo]
    processing_time_ms: float
    debug: Optional[Dict] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str


# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if state.is_ready else "loading",
        models_loaded=state.is_ready,
        version="1.0.0"
    )


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Main endpoint: Ask a question, get an AI-synthesized answer.
    
    Flow:
    1. Classify intent (or use forced mode)
    2. Retrieve relevant content from Wikipedia/Reddit
    3. Synthesize into a formatted answer
    4. Return with citations and confidence
    """
    if not state.is_ready:
        raise HTTPException(503, "Models are still loading. Please retry.")
    
    start_time = time.time()
    
    # Step 1: Intent Classification
    mode_was_forced = request.mode is not None
    
    if request.mode:
        detected_mode = request.mode.lower()
        mode_confidence = 1.0
        intent_result = None
    else:
        intent_result = state.classifier.predict(request.query)
        detected_mode = intent_result.predicted_mode
        mode_confidence = intent_result.confidence
        
        # If ambiguous, log it (great for improving the model later)
        if intent_result.is_ambiguous:
            logger.warning(
                f"Ambiguous intent for: '{request.query}' | "
                f"Top: {detected_mode} ({mode_confidence:.2f})"
            )
    
    logger.info(f"Query: '{request.query}' | Mode: {detected_mode} | Conf: {mode_confidence:.2f}")
    
    # Step 2: Retrieve relevant content
    retrieved_chunks = state.retrieval_pipeline.process_query(
        query=request.query,
        mode=detected_mode,
        top_k=request.top_k,
        use_mmr=True
    )
    
    # Step 3: Synthesize answer
    answer = state.synthesizer.synthesize(
        query=request.query,
        mode=detected_mode,
        retrieved_chunks=retrieved_chunks,
        max_sentences=12
    )
    
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Build response
    sources = [
        SourceInfo(
            index=c.index,
            title=c.title,
            url=c.url,
            source_type=c.source_type,
            relevance_score=round(c.relevance_score, 3)
        )
        for c in answer.citations
    ]
    
    debug_info = None
    if request.include_debug:
        debug_info = {
            'intent_probabilities': (
                intent_result.all_probabilities if intent_result else {}
            ),
            'chunks_retrieved': len(retrieved_chunks),
            'chunk_scores': [round(s, 3) for _, s in retrieved_chunks],
            'synthesis_metadata': answer.metadata
        }
    
    return QueryResponse(
        query=request.query,
        detected_mode=detected_mode,
        mode_confidence=round(mode_confidence, 3),
        mode_was_forced=mode_was_forced,
        answer=answer.answer_text,
        confidence=answer.confidence,
        key_points=answer.key_points,
        sources=sources,
        processing_time_ms=round(processing_time, 1),
        debug=debug_info
    )


@app.post("/feedback")
async def submit_feedback(query: str, was_helpful: bool, correct_mode: Optional[str] = None):
    """
    Collect user feedback for model improvement.
    
    This creates a feedback loop:
    - Wrong mode classifications get logged → retrain classifier
    - Unhelpful answers get logged → improve retrieval/synthesis
    
    INTERVIEW POINT: "I built a feedback loop for continuous improvement,
    which is how production ML systems actually work."
    """
    feedback_entry = {
        'query': query,
        'was_helpful': was_helpful,
        'correct_mode': correct_mode,
        'timestamp': time.time()
    }
    
    # In production, this would go to a database
    # For now, append to a JSONL file
    import json
    with open('data/feedback.jsonl', 'a') as f:
        f.write(json.dumps(feedback_entry) + '\n')
    
    logger.info(f"Feedback received: helpful={was_helpful}")
    return {"status": "feedback recorded"}


# ============================================================
# RUN: uvicorn app.main:app --reload --port 8000
# ============================================================
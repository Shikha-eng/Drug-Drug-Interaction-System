# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .schemas import DDIRequest, DDIResponse
from .model import predict, EMBEDDINGS, METADATA

app = FastAPI(
    title="Drug-Drug Interaction Predictor",
    description=(
        "Predicts harmful drug-drug interactions using Graph ML "
        "(Node2Vec + XGBoost) trained on DDInter data. "
        "AUROC: 0.9777 | AUPRC: 0.9757 | 1,641 drugs covered."
    ),
    version="1.0.0"
)

# Allow frontend to call this API from a browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- API routes must be declared BEFORE the static mount at "/" ----

@app.post("/predict", response_model=DDIResponse)
def predict_interaction(request: DDIRequest):
    if not request.drug_a or not request.drug_b:
        raise HTTPException(status_code=400, detail="Both drug names required")
    if request.drug_a.strip().lower() == request.drug_b.strip().lower():
        raise HTTPException(status_code=400, detail="Drug A and Drug B must be different")

    result = predict(request.drug_a, request.drug_b)
    return result

@app.get("/drugs")
def list_drugs():
    """Returns all drug names known to the model"""
    return {
        "total": len(EMBEDDINGS),
        "drugs": sorted(EMBEDDINGS.keys())
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost + Node2Vec",
        "drugs_in_graph": len(EMBEDDINGS),
        "auroc": METADATA["metrics"]["auroc"],
        "threshold": METADATA["optimal_threshold"]
    }

# ---- Static frontend mounted LAST, at root ----
# This serves frontend/index.html at "/" and any other static
# assets in the frontend folder. Must come after all API routes above,
# since it's a catch-all for unmatched paths.
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

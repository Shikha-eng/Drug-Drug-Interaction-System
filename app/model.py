# app/model.py
import pickle
import json
import numpy as np
from pathlib import Path

# Paths
BASE = Path(__file__).parent.parent / "data"

# Load everything once at startup — not on every request
with open(BASE / "xgb_final_model.pkl", "rb") as f:
    XGB_MODEL = pickle.load(f)

with open(BASE / "node2vec_embeddings.pkl", "rb") as f:
    EMBEDDINGS = pickle.load(f)

with open(BASE / "model_metadata.json", "r") as f:
    METADATA = json.load(f)

THRESHOLD = METADATA["optimal_threshold"]  # 0.4391
RISK_BUCKETS = METADATA["risk_buckets"]

def get_risk_level(probability: float) -> str:
    if probability >= RISK_BUCKETS["HIGH"][0]:
        return "INTERACTION LIKELY"
    elif probability >= RISK_BUCKETS["MEDIUM"][0]:
        return "POSSIBLE INTERACTION"
    else:
        return "UNLIKELY TO INTERACT"

def compute_edge_features(u_emb: np.ndarray, v_emb: np.ndarray) -> np.ndarray:
    hadamard = u_emb * v_emb
    l1 = np.abs(u_emb - v_emb)
    l2 = (u_emb - v_emb) ** 2
    avg = (u_emb + v_emb) / 2
    return np.concatenate([hadamard, l1, l2, avg])

def predict(drug_a: str, drug_b: str) -> dict:
    known_drugs = {d.lower(): d for d in EMBEDDINGS.keys()}
    
    drug_a_key = known_drugs.get(drug_a.strip().lower())
    drug_b_key = known_drugs.get(drug_b.strip().lower())
    
    both_known = drug_a_key is not None and drug_b_key is not None
    
    if not both_known:
        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "interaction_probability": None,
            "risk_level": "UNKNOWN",
            "both_drugs_known": False,
            "disclaimer": (
                "One or both drugs not found in the knowledge graph. "
                "This system covers 1,641 small-molecule drugs. "
                "Biologics, vaccines, and novel compounds are not supported."
            )
        }
    
    try:
        emb_a = EMBEDDINGS[drug_a_key]
        emb_b = EMBEDDINGS[drug_b_key]
        features = compute_edge_features(emb_a, emb_b).reshape(1, -1)
        probability = float(XGB_MODEL.predict_proba(features)[0][1])
        risk_level = get_risk_level(probability)
        
        return {
            "drug_a": drug_a_key,
            "drug_b": drug_b_key,
            "interaction_probability": round(probability, 4),
            "risk_level": risk_level,
            "both_drugs_known": True,
            "disclaimer": (
                "This tool is for research purposes only. "
                "Model has a 7.5% false negative rate — do not use as sole "
                "clinical decision support. Always consult a pharmacist or physician."
            )
        }
    except Exception as e:
        # Never return 500 to client — return structured error instead
        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "interaction_probability": None,
            "risk_level": "ERROR",
            "both_drugs_known": both_known,
            "disclaimer": f"Prediction failed internally: {str(e)}"
        }
# app/schemas.py
from pydantic import BaseModel
from typing import Optional

class DDIRequest(BaseModel):
    drug_a: str
    drug_b: str

    class Config:
        json_schema_extra = {
            "example": {
                "drug_a": "Warfarin",
                "drug_b": "Aspirin"
            }
        }

class DDIResponse(BaseModel):
    drug_a: str
    drug_b: str
    interaction_probability: float
    risk_level: str
    both_drugs_known: bool
    disclaimer: str
    model_note: str = "Predicts interaction probability only — not severity. Minor, Moderate, and Major interactions may all score HIGH probability."
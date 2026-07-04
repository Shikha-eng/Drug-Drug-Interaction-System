# DDI Predict — Drug-Drug Interaction Prediction using Graph ML

> A Graph ML system that predicts potential drug-drug interactions using **Node2Vec embeddings, XGBoost, RDKit molecular fingerprints, FastAPI, and Docker**.


## Overview

Drug-drug interactions (DDIs) can contribute to adverse drug events and patient harm. This project models DDI prediction as a **link prediction problem on a drug knowledge graph**:

- Drugs → graph nodes
- Known interactions → graph edges
- Prediction → probability that an interaction exists between two drugs

The system combines graph topology with molecular information and exposes predictions through a FastAPI backend and interactive frontend.

---

## Model Performance

| Metric | Score |
|---|---:|
| AUROC | **0.9777** |
| AUPRC | **0.9757** |
| Accuracy | **93.0%** |
| Precision | 0.93 |
| Recall | 0.93 |
| F1 Score | 0.93 |

**Graph size:** 1,641 drugs · 106,757 interaction edges

> The model predicts interaction probability, not clinical severity.

---

## Architecture

```text
DDInter Dataset
      ↓
Data Cleaning & Deduplication
      ↓
PubChem SMILES Retrieval
      ↓
RDKit Morgan Fingerprints
      ↓
Drug Knowledge Graph
      ↓
Node2Vec Embeddings (128-dim)
      ↓
Edge Feature Engineering
      ↓
XGBoost Classifier
      ↓
FastAPI + Frontend + Docker

ML Pipeline
1. Data Processing
Combined 8 DDInter ATC-category datasets
Deduplicated drug pairs
Retrieved SMILES from PubChem
Generated 2048-bit Morgan fingerprints using RDKit
2. Graph Learning
Built a NetworkX drug interaction graph
Trained 128-dimensional Node2Vec embeddings
Used a connectivity-aware train/test strategy to reduce leakage
3. Edge Features

For each drug pair, four symmetric operators were applied:

Hadamard product
L1 distance
L2 distance
Average

The resulting vectors were concatenated into a 512-dimensional edge representation.

4. Classification

An XGBoost classifier predicts the probability of an interaction between two drugs.

Tech Stack
Component	Technology
Language	Python
Graph Processing	NetworkX
Molecular Features	RDKit
Graph Embeddings	Node2Vec
ML Model	XGBoost
Explainability	SHAP
Backend	FastAPI + Uvicorn
Frontend	HTML, CSS, JavaScript, D3.js
Containerization	Docker
Data Sources	DDInter + PubChem
Project Structure
ddi_api/
├── app/
│   ├── main.py
│   ├── model.py
│   └── schemas.py
├── data/
│   └── processed/
├── frontend/
│   └── index.html
├── notebooks/
│   ├── phase1_data_pipeline.ipynb
│   ├── phase2_node2vec.ipynb
│   └── phase3_xgboost.ipynb
├── test_api.py
├── requirements.txt
├── Dockerfile
└── README.md
Quick Start with Docker
docker pull shikhaayad1508/ddipred:latest
docker run -p 8000:8000 shikhaayad1508/ddipred:latest

Then open:

Frontend: http://localhost:8000/ui
Swagger Docs: http://localhost:8000/docs
Health Check: http://localhost:8000/health
Local Setup

Clone the repository:

git clone https://github.com/shikhaayad1508/ddi-predict.git
cd ddi-predict

Install dependencies:

pip install -r requirements.txt

Run the API:

uvicorn app.main:app --reload --port 8000

Large trained model artifacts are excluded from GitHub due to file-size limits. Use the Docker image for the complete runnable system.

API Usage
Predict Interaction

POST /predict

Request:

{
  "drug_a": "Warfarin",
  "drug_b": "Ibuprofen"
}

Example response:

{
  "drug_a": "Warfarin",
  "drug_b": "Ibuprofen",
  "interaction_probability": 0.9832,
  "risk_level": "INTERACTION LIKELY",
  "both_drugs_known": true
}
Other Endpoints
Endpoint	Method	Description
/predict	POST	Predict interaction probability
/drugs	GET	List supported drugs
/health	GET	Check API and model status
/docs	GET	Swagger API documentation
Known Limitations
Predicts interaction existence, not clinical severity
Limited to drugs represented in the knowledge graph
Does not incorporate patient-specific factors
Biologics without suitable small-molecule SMILES representations are excluded
Intended for research and educational use only
Future Work
Multi-class severity prediction
GraphSAGE for unseen-drug generalization
RxNorm drug-class integration
Patient-specific risk features
Expanded drug and interaction coverage
Disclaimer

This project is for research and educational purposes only. It is not validated for clinical use and must not replace professional medical or pharmaceutical judgment.

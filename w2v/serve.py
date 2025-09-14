from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json, numpy as np, torch
import os

app = FastAPI(
    title="Word2Vec Neighbors API",
    description="Find similar words using trained Word2Vec embeddings",
    version="1.0.0"
)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
artifacts_dir = os.path.join(project_root, "artifacts")
static_dir = os.path.join(project_root, "static")

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

vecs = torch.load(os.path.join(artifacts_dir, "vectors.pt")).numpy()
with open(os.path.join(artifacts_dir, "vocab.json")) as f:
    vocab = json.load(f)
stoi = {k:int(v) for k,v in vocab["stoi"].items()}
itos = {int(k):v for k,v in vocab["itos"].items()}
norms = np.linalg.norm(vecs, axis=1) + 1e-9

@app.get("/")
def root():
    """Serve the web UI"""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/api")
def api_info():
    """API information endpoint"""
    return {
        "message": "Welcome to Word2Vec Neighbors API!",
        "description": "Find similar words using trained Word2Vec embeddings",
        "endpoints": {
            "/neighbors": "Find similar words (GET /neighbors?word=WORD&k=NUMBER)",
            "/vocab": "Get vocabulary information",
            "/docs": "Interactive API documentation",
            "/redoc": "Alternative API documentation"
        },
        "example": "/neighbors?word=neural&k=5"
    }

@app.get("/vocab")
def get_vocab_info():
    """Get vocabulary information"""
    return {
        "vocab_size": len(stoi),
        "sample_words": list(stoi.keys())[:20],
        "total_embeddings": vecs.shape[0],
        "embedding_dim": vecs.shape[1]
    }

@app.get("/neighbors")
def neighbors(word: str, k: int = 10):
    """Find the k most similar words to the given word"""
    if word not in stoi:
        raise HTTPException(status_code=404, detail=f"'{word}' not in vocab")
    idx = stoi[word]
    q = vecs[idx]
    sims = (vecs @ q) / (norms * (np.linalg.norm(q)+1e-9))
    order = np.argsort(-sims)
    out=[]
    for i in order:
        if i==idx: continue
        out.append({"word": itos.get(i,"?"), "cosine": float(sims[i])})
        if len(out)>=k: break
    return {"query": word, "neighbors": out}

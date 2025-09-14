import argparse, json
import numpy as np
import torch

def main(args):
    vecs = torch.load("artifacts/vectors.pt").numpy()
    with open("artifacts/vocab.json") as f:
        vocab = json.load(f)
    stoi = {k:int(v) for k,v in vocab["stoi"].items()}
    itos = {int(k):v for k,v in vocab["itos"].items()}
    if args.word not in stoi:
        print(f"Word '{args.word}' not in vocab")
        return
    idx = stoi[args.word]
    q = vecs[idx]
    # cosine similarity
    norms = np.linalg.norm(vecs, axis=1) * (np.linalg.norm(q) + 1e-9)
    sims = (vecs @ q) / (norms + 1e-9)
    topk = np.argsort(-sims)[:args.topk+1]
    for i in topk:
        if i==idx: continue
        print(f"{itos.get(i, '?'):20s}  cos={sims[i]:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    main(args)

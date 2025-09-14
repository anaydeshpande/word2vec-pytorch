import argparse, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main(args):
    vecs = torch.load("artifacts/vectors.pt").numpy()
    with open("artifacts/vocab.json") as f:
        vocab = json.load(f)
    itos = {int(k):v for k,v in vocab["itos"].items()}

    limit = min(args.limit, vecs.shape[0])
    X = vecs[:limit]
    labels = [itos.get(i, "?") for i in range(limit)]
    X2 = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="random", n_iter=1000).fit_transform(X)

    plt.figure(figsize=(12,10))
    plt.scatter(X2[:,0], X2[:,1], s=8)
    for i, txt in enumerate(labels):
        if i % max(1, limit//200) == 0:  # avoid overplotting
            plt.annotate(txt, (X2[i,0], X2[i,1]), fontsize=8, alpha=0.7)
    plt.title("t-SNE of Word2Vec embeddings")
    plt.tight_layout()
    out = "artifacts/tsne.png"
    plt.savefig(out, dpi=180)
    print(f"Saved {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()
    main(args)

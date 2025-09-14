import argparse, json, os, math
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from .data import SGNSDataset
from .model import SGNS

def collate(batch):
    centers, contexts, negs = zip(*batch)
    return torch.stack(centers), torch.stack(contexts), torch.stack(negs)

def main(args):
    ds = SGNSDataset(args.corpus, window_size=args.window, min_freq=args.min_freq,
                     max_vocab=args.max_vocab, subsample_t=args.subsample_t,
                     negatives=args.negatives)
    vocab_size = len(ds.itos)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SGNS(vocab_size, emb_dim=args.dim).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate, drop_last=True)

    steps = 0
    for epoch in range(1, args.epochs+1):
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        epoch_loss = 0
        batch_count = 0
        for centers, contexts, negs in pbar:
            centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
            loss = model(centers, contexts, negs)
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1
            epoch_loss += loss.item()
            batch_count += 1
            if steps % 200 == 0:
                pbar.set_postfix(loss=float(loss.item()))
        print(f"Epoch {epoch} loss: {epoch_loss/batch_count:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.get_input_vectors(), "artifacts/vectors.pt")
    # also export TSV for projector-like tools
    vecs = model.get_input_vectors().numpy()
    with open("artifacts/vocab.json") as f:
        vocab = json.load(f)["itos"]
    with open("artifacts/vectors.tsv","w") as vf, open("artifacts/meta.tsv","w") as mf:
        for i in range(vecs.shape[0]):
            vf.write("\t".join(str(x) for x in vecs[i]) + "\n")
            mf.write(vocab.get(str(i), "<unk>") + "\n")
    print("Saved embeddings to artifacts/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="data/sample.txt")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_vocab", type=int, default=50000)
    ap.add_argument("--subsample_t", type=float, default=1e-5)
    ap.add_argument("--negatives", type=int, default=5)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)

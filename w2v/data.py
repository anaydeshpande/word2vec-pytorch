import json, math, os, re, random
from collections import Counter
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"[A-Za-z0-9_#+\-']+")

def tokenize(text:str)->List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]

def build_vocab(tokens:List[str], min_freq:int=2, max_size:int=50000):
    freq = Counter(tokens)
    items = [(w,c) for w,c in freq.items() if c>=min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[:max_size-2]
    stoi = {"<unk>":0, "<pad>":1}
    for i,(w,_) in enumerate(items, start=2):
        stoi[w]=i
    itos = {i:w for w,i in stoi.items()}
    counts = np.zeros(len(stoi), dtype=np.float64)
    for w,c in freq.items():
        idx = stoi[w] if w in stoi else 0
        counts[idx] += c
    return stoi, itos, counts

def subsample(tokens:List[str], counts:np.ndarray, stoi:dict, t:float=1e-5):
    total = counts.sum()
    probs = {}
    for w,idx in stoi.items():
        f = counts[idx]/total if total>0 else 0
        if f>0:
            probs[w] = max(0.0, 1 - math.sqrt(t/f))
        else:
            probs[w] = 0.0
    kept=[]
    for w in tokens:
        if w in probs and random.random() < probs[w]:
            continue
        kept.append(w if w in stoi else "<unk>")
    return kept

class SGNSDataset(Dataset):
    """
    Produces (center, context, negatives) triples for SGNS.
    Uses a sliding window and a unigram^0.75 negative sampler.
    """
    def __init__(self, corpus_path:str, window_size:int=5, min_freq:int=2, max_vocab:int=50000,
                 subsample_t:float=1e-5, negatives:int=5, table_size:int=1_000_000):
        text = open(corpus_path, "r", encoding="utf-8").read()
        raw_tokens = tokenize(text)
        self.stoi, self.itos, counts = build_vocab(raw_tokens, min_freq, max_vocab)
        tokens = subsample(raw_tokens, counts, self.stoi, subsample_t)
        self.ids = [self.stoi.get(w,0) for w in tokens]
        self.window = window_size
        self.neg_k = negatives
        # Build negative sampling table
        pow_counts = counts ** 0.75
        self.neg_dist = pow_counts / pow_counts.sum() if pow_counts.sum()>0 else np.ones_like(pow_counts)/len(pow_counts)
        self.neg_table = np.random.choice(len(self.neg_dist), size=table_size, p=self.neg_dist)
        # Precompute center-context pairs (skip pads)
        self.pairs: List[Tuple[int,int]] = []
        for i, c in enumerate(self.ids):
            if c==1: continue
            left = max(0, i - self.window)
            right = min(len(self.ids), i + self.window + 1)
            for j in range(left, right):
                if j==i: continue
                ctx = self.ids[j]
                if ctx==1: continue
                self.pairs.append((c, ctx))
        # Save vocab
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/vocab.json","w") as f:
            json.dump({"stoi":self.stoi, "itos":self.itos}, f)

    def __len__(self):
        return len(self.pairs)

    def sample_negatives(self, exclude_idx:int, k:int)->np.ndarray:
        # Sample until we have k tokens that are not the positive context, allow duplicates
        negs = []
        i = 0
        while len(negs)<k and i < k*10:
            idx = int(self.neg_table[np.random.randint(0, len(self.neg_table))])
            if idx != exclude_idx and idx != 1:
                negs.append(idx)
            i += 1
        # Fallback if very small vocab
        while len(negs)<k:
            negs.append(0)
        return np.array(negs, dtype=np.int64)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        negs = self.sample_negatives(context, self.neg_k)
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long), torch.tensor(negs, dtype=torch.long)

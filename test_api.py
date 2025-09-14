#!/usr/bin/env python3
"""
Simple test script for the Word2Vec API functionality
"""
import sys
import os
sys.path.append('/Users/anay/word2vec-pytorch')

import json
import numpy as np
import torch

def test_word_neighbors(word, k=5):
    """Test the word neighbors functionality directly"""
    try:
        # Load the data (same as serve.py)
        vecs = torch.load('artifacts/vectors.pt').numpy()
        with open('artifacts/vocab.json') as f:
            vocab = json.load(f)
        stoi = {k:int(v) for k,v in vocab['stoi'].items()}
        itos = {int(k):v for k,v in vocab['itos'].items()}
        norms = np.linalg.norm(vecs, axis=1) + 1e-9
        
        if word not in stoi:
            print(f"❌ Word '{word}' not in vocabulary")
            return None
            
        idx = stoi[word]
        q = vecs[idx]
        sims = (vecs @ q) / (norms * (np.linalg.norm(q)+1e-9))
        order = np.argsort(-sims)
        
        neighbors = []
        for i in order:
            if i == idx: continue
            neighbors.append({"word": itos.get(i,'?'), "cosine": float(sims[i])})
            if len(neighbors) >= k: break
        
        return {"query": word, "neighbors": neighbors}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🧪 Testing Word2Vec API Functionality")
    print("=" * 50)
    
    # Test words
    test_words = ["neural", "learning", "machine", "data", "model"]
    
    for word in test_words:
        print(f"\n🔍 Testing word: '{word}'")
        result = test_word_neighbors(word, k=5)
        
        if result:
            print(f"✅ Query: {result['query']}")
            print("📊 Top neighbors:")
            for i, neighbor in enumerate(result['neighbors'], 1):
                print(f"   {i}. {neighbor['word']:15s} (cosine: {neighbor['cosine']:.4f})")
        else:
            print(f"❌ Failed to get neighbors for '{word}'")
    
    print("\n" + "=" * 50)
    print("🎯 API functionality test complete!")
    print("\n💡 To start the actual server, run:")
    print("   python3 -m uvicorn w2v.serve:app --port 8000 --host 127.0.0.1")
    print("   Then visit: http://127.0.0.1:8000/docs")

if __name__ == "__main__":
    main()

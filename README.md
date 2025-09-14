# Word2Vec PyTorch Implementation 🧠

A complete, production-ready implementation of Word2Vec using PyTorch, featuring Skip-gram with Negative Sampling (SGNS) as described in the original research papers by Mikolov et al.

## 📚 Research Foundation

This implementation is based on the seminal Word2Vec papers:

- **["Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/abs/1301.3781)** (Mikolov et al., 2013) - Introduced CBOW and Skip-gram models
- **["Distributed Representations of Words and Phrases and their Compositionality"](https://arxiv.org/abs/1310.4546)** (Mikolov et al., 2013, NIPS) - Introduced negative sampling, sub-sampling, and phrase vectors

## 🎯 What We Built

A complete Word2Vec pipeline from scratch, including:

- ✅ **Neural Network Implementation** - Skip-gram with Negative Sampling
- ✅ **Data Processing Pipeline** - Tokenization, vocabulary building, subsampling
- ✅ **Training System** - Full training loop with progress tracking
- ✅ **Evaluation Tools** - Nearest neighbors search and similarity analysis
- ✅ **Visualization** - t-SNE plots for embedding exploration
- ✅ **Web API** - FastAPI server for real-time queries
- ✅ **Beautiful Web UI** - Modern, responsive interface for word exploration

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python3 -m w2v.train --corpus data/sample.txt --epochs 5 --dim 128
```

### 3. Start the Web Interface
```bash
python3 -m uvicorn w2v.serve:app --port 8000 --host 127.0.0.1
```

Then open your browser to: **http://127.0.0.1:8000/**

## 🏗️ Architecture Overview

This implementation faithfully reproduces the original Word2Vec algorithm with modern Python best practices:

```
Text Corpus → Tokenization → Vocabulary → Skip-gram Training → Word Embeddings
     ↓              ↓            ↓              ↓                    ↓
  Raw Text    →  Tokens   →  Word IDs   →  Neural Network  →  Vector Space
```

## 📁 Project Structure

```
word2vec-pytorch/
├── data/
│   └── sample.txt              # Training corpus
├── w2v/                        # Core module
│   ├── __init__.py
│   ├── data.py                 # Data processing & dataset
│   ├── model.py                # SGNS neural network
│   ├── train.py                # Training script
│   ├── eval_neighbors.py       # Nearest neighbors evaluation
│   ├── visualize_tsne.py       # t-SNE visualization
│   └── serve.py                # FastAPI server + Web UI
├── static/
│   └── index.html              # Beautiful web interface
├── artifacts/                  # Generated outputs
│   ├── vectors.pt              # Learned embeddings
│   ├── vocab.json              # Vocabulary mappings
│   ├── vectors.tsv             # TSV format embeddings
│   ├── meta.tsv                # Word labels
│   └── tsne.png                # 2D visualization
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔧 Core Components Explained

### 1. **Data Processing** (`w2v/data.py`)
**What it does**: Transforms raw text into training data for the neural network.

**Key Functions**:
- `tokenize()`: Converts text to lowercase tokens using regex
- `build_vocab()`: Creates vocabulary with frequency filtering
- `subsample()`: Implements Mikolov's subsampling technique to reduce frequent words
- `SGNSDataset`: PyTorch Dataset that generates center-context pairs for training

**Why it matters**: This is where we prepare the data that the neural network will learn from. The subsampling technique is crucial for good embeddings.

### 2. **Neural Network** (`w2v/model.py`)
**What it does**: Implements the core Word2Vec algorithm - Skip-gram with Negative Sampling.

**Key Components**:
- `SGNS`: The main neural network class
- Input embeddings: Maps word IDs to vectors
- Output embeddings: Maps word IDs to context vectors
- Loss function: Binary cross-entropy with negative sampling

**The Math**: 
```
loss = -log σ(v_c · v_o) - Σ log σ(-v_c · v_neg)
```
Where `v_c` is the center word vector, `v_o` is the context word vector, and `v_neg` are negative sample vectors.

### 3. **Training** (`w2v/train.py`)
**What it does**: Orchestrates the entire training process and saves the results.

**Key Features**:
- Adam optimizer with configurable learning rate
- Progress bars showing training loss
- Saves embeddings in multiple formats (PyTorch, TSV, JSON)
- Automatic GPU/CPU detection

**What happens**: The model learns to predict context words from center words, which forces it to learn meaningful word representations.

### 4. **Evaluation** (`w2v/eval_neighbors.py`)
**What it does**: Tests the quality of learned embeddings by finding similar words.

**How it works**: Uses cosine similarity to find the most similar words to any given word.

**Example**: If you query "neural", it might return ["networks", "learning", "artificial", "intelligence"].

### 5. **Visualization** (`w2v/visualize_tsne.py`)
**What it does**: Creates 2D visualizations of the high-dimensional word embeddings.

**The Process**: Uses t-SNE to reduce 128-dimensional vectors to 2D for plotting.

**Why it's useful**: You can visually see how similar words cluster together in the embedding space.

### 6. **Web Server** (`w2v/serve.py`)
**What it does**: Provides a REST API and beautiful web interface for querying embeddings.

**Endpoints**:
- `/` - Beautiful web UI for word exploration
- `/neighbors?word=X&k=Y` - API endpoint for finding similar words
- `/vocab` - Vocabulary information
- `/docs` - Interactive API documentation

## 🎨 Web Interface Features

The web UI (`static/index.html`) provides:

- **🔍 Interactive Search**: Type any word and see similar words instantly
- **📊 Similarity Scores**: See how similar each word is (as a percentage)
- **⚙️ Customizable Results**: Choose 3, 5, 10, or 15 similar words
- **💡 Example Words**: Click on example words to quickly test
- **📱 Responsive Design**: Works on desktop, tablet, and mobile
- **🎨 Modern UI**: Beautiful gradient design with smooth animations

## 🔬 Algorithm Deep Dive

### Skip-gram Model
Instead of predicting a word from its context (like CBOW), Skip-gram predicts context words from a center word. This is more effective for learning word representations.

### Negative Sampling
Instead of using the expensive hierarchical softmax, we sample negative examples and use binary classification. This makes training much faster.

### Subsampling
Frequent words like "the", "and", "of" are downsampled because they don't provide much semantic information. The probability of keeping a word is:
```
P(keep) = 1 - sqrt(t / f)
```
Where `t` is a threshold (usually 1e-5) and `f` is the word frequency.

## 📊 Generated Artifacts

After training, you get:

- **`vectors.pt`**: PyTorch tensor with learned embeddings
- **`vocab.json`**: Word-to-ID and ID-to-word mappings
- **`vectors.tsv`**: Embeddings in TSV format (for TensorBoard Projector)
- **`meta.tsv`**: Word labels for the TSV file
- **`tsne.png`**: 2D visualization of embeddings

## 🚀 Usage Examples

### Training with Custom Parameters
```bash
python3 -m w2v.train \
    --corpus data/my_corpus.txt \
    --epochs 10 \
    --dim 256 \
    --window 10 \
    --lr 0.001
```

### Command Line Evaluation
```bash
# Find similar words
python3 -m w2v.eval_neighbors --word intelligence --topk 15

# Create visualization
python3 -m w2v.visualize_tsne --limit 1000
```

### API Usage
```bash
# Start server
python3 -m uvicorn w2v.serve:app --port 8000

# Query via API
curl "http://localhost:8000/neighbors?word=artificial&k=5"

# Or use the beautiful web interface
open http://localhost:8000/
```

## 🎓 Educational Value

This implementation is perfect for understanding:

- **Word Embeddings**: How neural networks learn word representations
- **PyTorch**: Modern deep learning framework usage
- **NLP Pipelines**: Complete text processing workflows
- **Web APIs**: Building production-ready services
- **Data Visualization**: Making high-dimensional data understandable

## 🔧 Technical Highlights

- **Efficient Training**: Negative sampling for fast convergence
- **Memory Optimized**: Precomputed sampling tables
- **GPU Support**: Automatic CUDA detection
- **Production Ready**: FastAPI with proper error handling
- **Multiple Formats**: Flexible output options
- **Modern UI**: Responsive web interface

## 📚 Dependencies

- **PyTorch**: Neural network framework
- **NumPy**: Numerical computations
- **FastAPI**: Web API framework
- **scikit-learn**: t-SNE visualization
- **Matplotlib**: Plotting
- **tqdm**: Progress bars
- **uvicorn**: ASGI server

## 🎯 Key Insights

1. **Word2Vec works by predicting context**: The model learns that words appearing in similar contexts should have similar representations.

2. **Negative sampling is crucial**: It makes training efficient by focusing on a few negative examples instead of the entire vocabulary.

3. **Subsampling improves quality**: Removing frequent words helps the model focus on meaningful semantic relationships.

4. **Dual embeddings**: Having separate input and output embeddings often works better than sharing weights.

5. **Visualization reveals structure**: t-SNE plots show how the model organizes words in semantic clusters.

## 🏆 What Makes This Special

This isn't just a toy implementation - it's a complete, production-ready system that:

- ✅ Faithfully implements the original Word2Vec algorithm
- ✅ Includes a beautiful, modern web interface
- ✅ Provides both API and UI access
- ✅ Generates multiple output formats
- ✅ Has comprehensive documentation
- ✅ Uses modern Python best practices
- ✅ Is ready for deployment

Perfect for learning, research, or as a foundation for larger NLP projects!

---

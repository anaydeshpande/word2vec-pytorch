# W2V Module Documentation

This module contains the core implementation of Word2Vec using Skip-gram with Negative Sampling (SGNS). Each file serves a specific purpose in the word embedding pipeline.

## 📁 Module Structure

```
w2v/
├── __init__.py          # Module initialization
├── data.py              # Data processing and dataset classes
├── model.py             # Neural network model definition
├── train.py             # Training script and utilities
├── eval_neighbors.py    # Nearest neighbors evaluation
├── visualize_tsne.py    # t-SNE visualization
└── serve.py             # FastAPI web server
```

## 🔧 Component Overview

### 1. Data Processing (`data.py`)
**Purpose**: Handles text preprocessing, vocabulary building, and dataset creation for training.

**Key Components**:
- `tokenize()`: Regex-based text tokenization
- `build_vocab()`: Creates vocabulary with frequency filtering
- `subsample()`: Implements Mikolov's subsampling technique
- `SGNSDataset`: PyTorch Dataset for Skip-gram training

**Key Features**:
- Configurable vocabulary size and minimum frequency
- Subsampling of high-frequency words
- Negative sampling table precomputation
- Center-context pair generation

### 2. Neural Network (`model.py`)
**Purpose**: Defines the Skip-gram with Negative Sampling neural network architecture.

**Key Components**:
- `SGNS`: Main neural network class
- Input and output embedding matrices
- Forward pass with positive and negative sampling

**Key Features**:
- Dual embedding approach (input/output vectors)
- Efficient negative sampling loss computation
- Proper weight initialization
- GPU/CPU compatibility

### 3. Training (`train.py`)
**Purpose**: Orchestrates the training process and saves model artifacts.

**Key Components**:
- `main()`: Training loop with progress tracking
- `collate()`: Batch collation function
- Command-line argument parsing

**Key Features**:
- Adam optimizer with configurable learning rate
- Progress bars with loss tracking
- Multiple output formats (PyTorch, TSV, JSON)
- Automatic device detection (GPU/CPU)

### 4. Evaluation (`eval_neighbors.py`)
**Purpose**: Evaluates trained embeddings by finding nearest neighbors.

**Key Components**:
- `main()`: Command-line evaluation interface
- Cosine similarity computation
- Top-k neighbor retrieval

**Key Features**:
- Cosine similarity-based ranking
- Command-line interface
- Error handling for missing words
- Formatted output display

### 5. Visualization (`visualize_tsne.py`)
**Purpose**: Creates 2D visualizations of high-dimensional embeddings.

**Key Components**:
- `main()`: t-SNE visualization pipeline
- Matplotlib plotting with word labels
- Configurable visualization parameters

**Key Features**:
- t-SNE dimensionality reduction
- Scatter plot with word annotations
- Configurable point limits
- High-resolution output

### 6. API Server (`serve.py`)
**Purpose**: Provides a REST API for real-time embedding queries.

**Key Components**:
- FastAPI application setup
- `/neighbors` endpoint
- Precomputed similarity matrices

**Key Features**:
- FastAPI framework with automatic docs
- Real-time similarity queries
- JSON response format
- Error handling and validation

## 🚀 Usage Patterns

### Training Pipeline
```python
# 1. Create dataset
dataset = SGNSDataset("corpus.txt", window_size=5)

# 2. Initialize model
model = SGNS(vocab_size=len(dataset.itos), emb_dim=128)

# 3. Train with DataLoader
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
```

### Evaluation Pipeline
```python
# 1. Load trained embeddings
vectors = torch.load("artifacts/vectors.pt")

# 2. Compute similarities
similarities = cosine_similarity(vectors[word_idx], vectors)

# 3. Find top-k neighbors
top_k = np.argsort(-similarities)[:k]
```

### API Pipeline
```python
# 1. Start server
uvicorn w2v.serve:app --port 8000

# 2. Query embeddings
response = requests.get("http://localhost:8000/neighbors?word=learning&k=5")
```

## 🔬 Technical Details

### Data Flow
1. **Text → Tokens**: Regex tokenization with lowercase conversion
2. **Tokens → Vocabulary**: Frequency counting and filtering
3. **Vocabulary → Dataset**: Center-context pair generation
4. **Dataset → Model**: Skip-gram training with negative sampling
5. **Model → Embeddings**: Learned word representations
6. **Embeddings → Applications**: Similarity search, visualization, API

### Key Algorithms
- **Skip-gram**: Predicts context words from center words
- **Negative Sampling**: Efficient alternative to hierarchical softmax
- **Subsampling**: Reduces impact of frequent words
- **t-SNE**: Non-linear dimensionality reduction for visualization

### Performance Optimizations
- Precomputed negative sampling tables
- Efficient batch processing
- GPU acceleration support
- Memory-efficient data structures

## 📊 Output Formats

The module generates several output formats:

1. **PyTorch Tensors** (`.pt`): Native PyTorch format for continued training
2. **TSV Files** (`.tsv`): Tab-separated values for external tools
3. **JSON Files** (`.json`): Human-readable vocabulary mappings
4. **PNG Images** (`.png`): t-SNE visualizations
5. **REST API**: Real-time JSON responses

## 🎯 Design Principles

- **Modularity**: Each file has a single responsibility
- **Configurability**: Extensive hyperparameter control
- **Efficiency**: Optimized for both training and inference
- **Usability**: Clear interfaces and error handling
- **Extensibility**: Easy to modify and extend functionality

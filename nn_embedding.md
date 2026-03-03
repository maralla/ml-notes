## Overview

One-Hot Encoding + Linear Transform = Embedding

## Definition

For $N$ categories and embedding dimension $d$:

$$\text{Embedding}: \{0, 1, ..., N-1\} \rightarrow \mathbb{R}^d$$

$$\text{emb}(i) = W[i]$$

Where:
- $W \in \mathbb{R}^{N \times d}$ is the learnable weight matrix
- $i$ is the category ID (integer)
- $W[i]$ is row $i$ of the weight matrix (no computation, just lookup)

### Gradient Update

During backpropagation, only the accessed row is updated:

$$W[i] \leftarrow W[i] - \eta \cdot \nabla_{W[i]} \mathcal{L}$$

Where:
- $\eta$ is learning rate
- $\nabla_{W[i]} \mathcal{L}$ is gradient of loss w.r.t. embedding $i$
- Other rows remain unchanged

## Some Use Cases

1. Natural Language Processing (NLP)
```python
# Word embeddings - map words to vectors
vocab_size = 50000
embedding_dim = 300
word_embedding = nn.Embedding(vocab_size, embedding_dim)

# "cat" (id=1234) → [0.23, -0.45, 0.67, ..., 0.12]
# "dog" (id=5678) → [0.19, -0.42, 0.71, ..., 0.08]
# Similar words get similar vectors through training
```

50,000 words → 300 dimensions (167× compression).

2. Recommendation Systems
```python
# User and item embeddings
num_users = 1000000
num_items = 500000
embedding_dim = 128

user_embedding = nn.Embedding(num_users, embedding_dim)
item_embedding = nn.Embedding(num_items, embedding_dim)

# Predict rating: similarity(user_vec, item_vec)
```

Learn user preferences and item characteristics jointly.

## When to Use Embeddings

- Input is categorical/discrete (small fixed set of IDs)
- Need to learn relationships between categories
- Have many categories (compression benefit)
- IDs have no inherent ordering or magnitude

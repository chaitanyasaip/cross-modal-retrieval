import faiss
import numpy as np
import torch

class EmbeddingIndex:
    def __init__(self, embedding_dimension):
        self.index = faiss.IndexFlatL2(embedding_dimension) # Create a flat L2 index for efficient search

    def add_embeddings(self, embeddings):
        embeddings = np.vstack(embeddings).astype('float32') # Convert embeddings to numpy array and add batch dimension
        self.index.add(embeddings) # Add embeddings to index

    def query(self, embedding, k=5):
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy().astype('float32') # Convert embedding to numpy array
        else:
            embedding = embedding.astype('float32')
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1) # Ensure embedding has shape (1, embedding_dimension)
        distances, indices = self.index.search(embedding, k) # Search for k nearest neighbors
        return distances, indices
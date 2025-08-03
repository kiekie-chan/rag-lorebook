import faiss
import numpy as np
import os


class FaissEmbed:
    def __init__(self, dim: int, embeddings_dir: str, index_path='faiss.index'):
        os.makedirs(embeddings_dir, exist_ok=True)

        self.dim = dim
        self.index_path = os.path.join(embeddings_dir, index_path)
        self.index = faiss.IndexFlatL2(dim)

    def add(self, vec: np.ndarray):
        vec = vec.astype('float32')
        vec /= np.linalg.norm(vec)
        self.index.add(vec[np.newaxis, :])

    def save(self):
        faiss.write_index(self.index, self.index_path)

    def load(self):
        self.index = faiss.read_index(self.index_path)

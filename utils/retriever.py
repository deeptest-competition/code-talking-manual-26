import os
from typing import Callable

import faiss
import numpy as np
import tqdm

from utils.manual import load_chunks_from_directory
from config import get_config

config = get_config()

TOP_K = config["retriever"].get("top_k")
EMBEDDING_TYPE = config["retriever"].get("embedding_type")

def get_embedding_func(embedding_type: str) -> Callable:
    """
    Dynamically import and return the embedding function based on config.
    """
    if embedding_type == "local":
        from llm.utils.embeddings_local import get_embedding 
    else:
        from llm.utils.embeddings_openai import get_embedding
    return get_embedding
    
class Retriever:
    emb_path = "embeddings.npy"
    index_path = "faiss.index"

    def __init__(
        self, manual_path: str, embedding_type: str = "local"
    ):
        self.manual_path = manual_path
        self.embedding_fnc = get_embedding_func(embedding_type)

        chunks, texts, metadata = load_chunks_from_directory(manual_path)
        if os.path.exists(self.emb_path):
            print("[Retriever] Loading embeddings from disk...")
            emb_matrix = np.load(self.emb_path)
        else:
            print("[Retriever] Generating embeddings for", len(texts), "chunks...")
            response = []
            for text in tqdm.tqdm(texts):
                response.append(self.embedding_fnc(text))
            emb_matrix = np.array(response).astype("float32")
            np.save(self.emb_path, emb_matrix)
            print("[Retriever] Embeddings saved to disk.")
        dim = emb_matrix.shape[1]

        if os.path.exists(self.index_path):
            print("[Retriever] Loading FAISS index from disk...")
            index = faiss.read_index(self.index_path)
        else:
            print("[Retriever] Building FAISS index...")
            index = faiss.IndexFlatL2(dim)
            index.add(emb_matrix)
            faiss.write_index(index, self.index_path)
            print("[Retriever] FAISS index built with", index.ntotal, "vectors and saved to disk.")
        self.index = index
        self.metadata = metadata
        self.texts = texts

    def retrieve_info(self, query, top_k=TOP_K):
        query_emb = self.embedding_fnc(query).astype("float32").reshape(1, -1)
        D, I = self.index.search(query_emb, top_k)
        return [self.metadata[i] | {"content": self.texts[i]} for i in I[0]]

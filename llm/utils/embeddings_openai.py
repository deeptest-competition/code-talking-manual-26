import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llm.llm_openai import get_openai_client
from config import get_config

# Load a pre-trained model from Hugging Face (e.g., a BERT-based model)
model_name = get_config()["embeddings"]["openai_model_name"]
threshold_similarity = get_config()["embeddings"]["similarity_threshold"]
model_api = get_config()["embeddings"]["model_api"]

def get_embedding(text, model=model_name, model_api = model_api):
    """Fetches the embedding vector for a given text using OpenAI's API."""
    client = get_openai_client(model_api)
    response = client.embeddings.create(input=text,
                                        model=model)
    return np.array(response.data[0].embedding)

def is_equal(a, b, threshold = threshold_similarity):
    reference_embedding = get_embedding(a).reshape(1, -1)  # Correct reshaping
    embedding = get_embedding(b).reshape(1, -1)  # Correct reshaping
    score = cosine_similarity(reference_embedding, embedding)[0][0]
    return score > threshold

def get_similarity(a,b, scale = None):
    reference_embedding = get_embedding(a).reshape(1, -1)  # Correct reshaping
    embedding = get_embedding(b).reshape(1, -1)  # Correct reshaping
    score = cosine_similarity(reference_embedding, embedding)[0][0]
    if scale is not None:
        score = score/scale
    print(score)
    return score
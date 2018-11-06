import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from embedding import VarTextEmbedder


def cluster_embeddings(varTextArray: np.ndarray, centroids: int) -> np.ndarray:
    embeddings = embed(varTextArray)

    kmeans = SemanticKMeans(centroids)
    predictions = kmeans.fit_predict(embeddings).reshape(-1, 1)

    return predictions


def embed(varTextArray: np.ndarray) -> np.ndarray:
    with VarTextEmbedder() as embedder:
        embeddings = embedder.embed(varTextArray)
    return embeddings


class SemanticKMeans:

    def __init__(self, centroids):
        self.num_centroids = centroids

    def fit(self, vectors: np.ndarray, max_iterations: int = 100) -> 'SemanticKMeans':
        centroids = np.random.uniform(-1., 1.,
                                      (self.num_centroids, vectors.shape[1]))

        for _ in range(max_iterations):
            distances = cosine_similarity(vectors, centroids).clip(-1., 1.)
            prev_centroids = np.copy(centroids)
            for c in range(self.num_centroids):
                members = vectors[np.argmax(distances, axis=1) == c]
                if len(members) > 0:
                    centroids[c] = np.mean(members, axis=0)

            if np.allclose(centroids, prev_centroids):
                break
        self.centroids = centroids
        return self

    def centroids_(self):
        return self.centroids

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        distances = cosine_similarity(vectors, self.centroids)
        return np.argmax(distances, axis=1)

    def fit_predict(self, vectors: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        return self.fit(vectors, max_iterations).predict(vectors)

import random
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


class Phase2Checker:
    """
    階段二：使用 Embedding（向量化）評估多樣性
    Phase 2: Vector space dispersion, coverage, and diversity evaluation.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._st_model = None
        self._init_model()

    def _init_model(self):
        """Try to load sentence-transformers model if available."""
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
        except Exception:
            self._st_model = None

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to dense vector array.
        Falls back to TF-IDF vectorizer if sentence-transformers is not available.
        """
        if self._st_model is not None:
            try:
                embeddings = self._st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                return embeddings
            except Exception:
                pass

        # Fallback to TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=256, stop_words=None)
        sparse_matrix = vectorizer.fit_transform(texts)
        return sparse_matrix.toarray()

    def evaluate_diversity(self, texts: List[str], sample_size: int = 500, random_seed: int = 42) -> Dict[str, Any]:
        """
        Evaluate text diversity across sample embeddings.
        """
        clean_texts = [str(t).strip() for t in texts if pd_not_na(t)]
        if len(clean_texts) == 0:
            return {"error": "No valid text entries"}

        # Random sampling if corpus is large
        if len(clean_texts) > sample_size:
            random.seed(random_seed)
            sampled_texts = random.sample(clean_texts, sample_size)
        else:
            sampled_texts = clean_texts

        embeddings = self.get_embeddings(sampled_texts)

        # 1. Centroid and dispersion
        centroid = np.mean(embeddings, axis=0)
        distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
        mean_dispersion = float(np.mean(distances_to_centroid))
        variance_dispersion = float(np.var(distances_to_centroid))

        # 2. Pairwise Cosine Distance
        sim_matrix = cosine_similarity(embeddings)
        # Extract upper triangle indices without diagonal
        triu_indices = np.triu_indices_from(sim_matrix, k=1)
        pairwise_similarities = sim_matrix[triu_indices]
        pairwise_distances = 1.0 - pairwise_similarities

        mean_pairwise_distance = float(np.mean(pairwise_distances)) if len(pairwise_distances) > 0 else 0.0
        avg_cosine_similarity = float(np.mean(pairwise_similarities)) if len(pairwise_similarities) > 0 else 1.0

        # 3. PCA Coverage (Explained Variance Ratio)
        n_components = min(10, embeddings.shape[0], embeddings.shape[1])
        if n_components > 1:
            pca = PCA(n_components=n_components)
            pca.fit(embeddings)
            explained_variance_ratio = pca.explained_variance_ratio_.tolist()
            top3_variance = float(sum(explained_variance_ratio[:3]))
        else:
            explained_variance_ratio = [1.0]
            top3_variance = 1.0

        # Diversity score calculation (0 - 100)
        # High pairwise distance & high mean dispersion = high diversity score
        diversity_score = min(100.0, max(0.0, (mean_pairwise_distance * 70.0) + (mean_dispersion * 30.0)))
        if self._st_model is None:
            # Scale score for TF-IDF fallback which naturally has higher sparsity/distance
            diversity_score = round(min(100.0, diversity_score * 0.95), 2)
        else:
            diversity_score = round(diversity_score, 2)

        return {
            "sample_count": len(sampled_texts),
            "vector_dimension": embeddings.shape[1],
            "embedding_method": "sentence-transformers" if self._st_model is not None else "tfidf_fallback",
            "mean_pairwise_distance": round(mean_pairwise_distance, 4),
            "avg_cosine_similarity": round(avg_cosine_similarity, 4),
            "mean_dispersion_from_centroid": round(mean_dispersion, 4),
            "top3_pca_explained_variance": round(top3_variance, 4),
            "diversity_score": diversity_score,
            "interpretation": (
                "High diversity & comprehensive topic coverage" if diversity_score >= 65
                else "Moderate diversity" if diversity_score >= 40
                else "Low diversity (high similarity / repetitive content)"
            )
        }


def pd_not_na(val):
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    return True

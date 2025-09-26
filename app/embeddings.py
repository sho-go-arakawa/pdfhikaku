"""Embedding calculation with support for multiple providers."""

import os
import hashlib
import pickle
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import time

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .config import config

logger = logging.getLogger(__name__)

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass

class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider (local)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise ImportError("sentence-transformers not available")

        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {self.model_name}: {e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence transformers."""
        if not self.model:
            raise RuntimeError("Model not loaded")

        # Filter out empty texts
        valid_texts = [text if text else " " for text in texts]

        return self.model.encode(valid_texts, convert_to_numpy=True)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider (requires API key)."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """Setup OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        if not HAS_OPENAI:
            logger.error("openai not installed. Install with: pip install openai")
            raise ImportError("openai not available")

        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI client with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        # Filter out empty texts
        valid_texts = [text if text else " " for text in texts]

        try:
            response = self.client.embeddings.create(
                input=valid_texts,
                model=self.model_name
            )

            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def get_dimension(self) -> int:
        """Get embedding dimension based on model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model_name, 1536)

class DummyEmbeddingProvider(EmbeddingProvider):
    """Dummy embedding provider that uses simple text features when advanced libraries are not available."""

    def __init__(self):
        logger.info("Using dummy embedding provider (no advanced NLP libraries)")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Create simple feature vectors from text."""
        embeddings = []

        for text in texts:
            # Create simple features: length, word count, character distribution
            text_lower = text.lower()
            features = [
                len(text),  # Text length
                len(text.split()),  # Word count
                text_lower.count('a') / max(len(text), 1),  # Character frequency features
                text_lower.count('e') / max(len(text), 1),
                text_lower.count('i') / max(len(text), 1),
                text_lower.count('o') / max(len(text), 1),
                text_lower.count('u') / max(len(text), 1),
                text.count('.') / max(len(text), 1),  # Punctuation features
                text.count(',') / max(len(text), 1),
                text.count(' ') / max(len(text), 1),
                # Japanese character features
                sum(1 for c in text if '\u3040' <= c <= '\u309F') / max(len(text), 1),  # Hiragana
                sum(1 for c in text if '\u30A0' <= c <= '\u30FF') / max(len(text), 1),  # Katakana
                sum(1 for c in text if '\u4E00' <= c <= '\u9FAF') / max(len(text), 1),  # Kanji
                # Numeric features
                sum(1 for c in text if c.isdigit()) / max(len(text), 1),
                sum(1 for c in text if c.isupper()) / max(len(text), 1),
                sum(1 for c in text if c.islower()) / max(len(text), 1),
            ]

            # Pad or truncate to fixed size
            target_size = 16
            if len(features) > target_size:
                features = features[:target_size]
            else:
                features.extend([0.0] * (target_size - len(features)))

            embeddings.append(features)

        return np.array(embeddings, dtype=np.float32)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return 16

class EmbeddingManager:
    """Manages embedding calculation with caching and batch processing."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or config.get_align_config()
        self.cache_dir = Path(config.get_perf_config()['cache_dir']) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize provider
        if self.cfg['use_openai'] and self.cfg['openai_api_key']:
            self.provider = OpenAIProvider(
                model_name=self.cfg['openai_model'],
                api_key=self.cfg['openai_api_key']
            )
        elif HAS_SENTENCE_TRANSFORMERS:
            self.provider = SentenceTransformerProvider(
                model_name=self.cfg['embedding_model']
            )
        else:
            # Use dummy provider if no embedding libraries available
            logger.warning("No embedding providers available, using dummy provider")
            self.provider = DummyEmbeddingProvider()

        self.batch_size = config.get_perf_config()['chunk_batch_size']

    def get_embeddings(self, texts: List[str], cache_key: Optional[str] = None) -> np.ndarray:
        """Get embeddings for texts with caching."""
        if not texts:
            return np.array([])

        # Generate cache key
        if cache_key is None:
            text_hash = hashlib.md5(''.join(texts).encode()).hexdigest()
            cache_key = f"{self.provider.__class__.__name__}_{self.provider.model_name}_{text_hash}"

        cache_path = self.cache_dir / f"{cache_key}.pkl"

        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if len(cached_data) == len(texts):
                        logger.info(f"Loaded embeddings from cache: {cache_path}")
                        return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")

        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} texts")
        start_time = time.time()

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.provider.encode(batch_texts)
            all_embeddings.append(batch_embeddings)

            if len(all_embeddings) % 10 == 0:  # Log progress every 10 batches
                logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")

        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])

        # Cache the results
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Cached embeddings to: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")

        elapsed = time.time() - start_time
        logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.2f} seconds")

        return embeddings

    def calculate_similarity_matrix(self, embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix between two sets of embeddings."""
        if embeddings_a.size == 0 or embeddings_b.size == 0:
            return np.array([])

        # Normalize embeddings
        norm_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        norm_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

        # Calculate cosine similarity
        similarity_matrix = np.dot(norm_a, norm_b.T)

        return similarity_matrix

    def find_best_matches(self, similarity_matrix: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find best matches from similarity matrix using Hungarian algorithm."""
        if similarity_matrix.size == 0:
            return []

        if not HAS_SCIPY:
            logger.warning("scipy not available, using greedy matching")
            return self._greedy_matching(similarity_matrix, threshold)

        try:
            # Convert similarity to cost (for minimization)
            cost_matrix = 1 - similarity_matrix

            # Apply threshold
            cost_matrix[similarity_matrix < threshold] = np.inf

            # Solve assignment problem
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matches = []
            for i, j in zip(row_indices, col_indices):
                if similarity_matrix[i, j] >= threshold:
                    matches.append({
                        'index_a': i,
                        'index_b': j,
                        'similarity': similarity_matrix[i, j]
                    })

            return matches

        except Exception as e:
            logger.warning(f"scipy matching failed: {e}, using greedy matching")
            return self._greedy_matching(similarity_matrix, threshold)

    def _greedy_matching(self, similarity_matrix: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        """Greedy matching as fallback when scipy is not available."""
        matches = []
        used_b = set()

        for i in range(similarity_matrix.shape[0]):
            best_j = -1
            best_score = threshold

            for j in range(similarity_matrix.shape[1]):
                if j in used_b:
                    continue

                score = similarity_matrix[i, j]
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j != -1:
                matches.append({
                    'index_a': i,
                    'index_b': best_j,
                    'similarity': best_score
                })
                used_b.add(best_j)

        return matches

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.provider.get_dimension()

    def clear_cache(self):
        """Clear embedding cache."""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared embedding cache")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

# Global embedding manager instance
embedding_manager = EmbeddingManager()
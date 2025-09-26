"""Section mapping and paragraph alignment using similarity matching."""

import logging
import numpy as np
import rapidfuzz
from typing import List, Dict, Any, Optional, Tuple
import time

from .embeddings import embedding_manager
from .config import config

logger = logging.getLogger(__name__)

class SectionMapper:
    """Maps sections between two documents using title similarity."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or config.get_align_config()
        self.title_weight = self.cfg['title_weight']
        self.embed_weight = self.cfg['embed_weight']
        self.string_weight = self.cfg['string_weight']
        self.exact_threshold = self.cfg['exact_threshold']
        self.partial_threshold = self.cfg['partial_threshold']

    def map_sections(self, sections_a: List[Dict[str, Any]],
                    sections_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create section mapping between two document structures."""
        logger.info(f"Mapping {len(sections_a)} sections from A to {len(sections_b)} sections from B")

        if not sections_a or not sections_b:
            return []

        # Extract titles for similarity calculation
        titles_a = [section['title'] for section in sections_a]
        titles_b = [section['title'] for section in sections_b]

        # Calculate similarity matrix
        similarity_matrix = self._calculate_title_similarity_matrix(titles_a, titles_b)

        # Find best matches using Hungarian algorithm or greedy matching
        matches = embedding_manager.find_best_matches(
            similarity_matrix, threshold=self.partial_threshold
        )

        # Create mapping objects
        mappings = []
        for match in matches:
            section_a = sections_a[match['index_a']]
            section_b = sections_b[match['index_b']]

            mapping = {
                'section_a_id': section_a['id'],
                'section_b_id': section_b['id'],
                'section_a_title': section_a['title'],
                'section_b_title': section_b['title'],
                'section_a_level': section_a['level'],
                'section_b_level': section_b['level'],
                'similarity_score': float(match['similarity']),
                'match_type': self._classify_match_type(match['similarity']),
                'section_a_data': section_a,
                'section_b_data': section_b
            }

            mappings.append(mapping)

        # Add unmatched sections
        matched_a = {m['index_a'] for m in matches}
        matched_b = {m['index_b'] for m in matches}

        for i, section in enumerate(sections_a):
            if i not in matched_a:
                mappings.append({
                    'section_a_id': section['id'],
                    'section_b_id': None,
                    'section_a_title': section['title'],
                    'section_b_title': None,
                    'section_a_level': section['level'],
                    'section_b_level': None,
                    'similarity_score': 0.0,
                    'match_type': 'deleted',
                    'section_a_data': section,
                    'section_b_data': None
                })

        for i, section in enumerate(sections_b):
            if i not in matched_b:
                mappings.append({
                    'section_a_id': None,
                    'section_b_id': section['id'],
                    'section_a_title': None,
                    'section_b_title': section['title'],
                    'section_a_level': None,
                    'section_b_level': section['level'],
                    'similarity_score': 0.0,
                    'match_type': 'added',
                    'section_a_data': None,
                    'section_b_data': section
                })

        # Sort by similarity score (descending)
        mappings.sort(key=lambda x: x['similarity_score'], reverse=True)

        logger.info(f"Created {len(mappings)} section mappings")
        return mappings

    def _calculate_title_similarity_matrix(self, titles_a: List[str],
                                         titles_b: List[str]) -> np.ndarray:
        """Calculate similarity matrix between two sets of titles."""
        # Combine string similarity and embedding similarity
        string_similarity = self._calculate_string_similarity_matrix(titles_a, titles_b)

        # Get embeddings if enabled
        if self.embed_weight > 0:
            try:
                embeddings_a = embedding_manager.get_embeddings(titles_a, "titles_a")
                embeddings_b = embedding_manager.get_embeddings(titles_b, "titles_b")
                embedding_similarity = embedding_manager.calculate_similarity_matrix(
                    embeddings_a, embeddings_b
                )
            except Exception as e:
                logger.warning(f"Failed to compute embeddings, using string similarity only: {e}")
                embedding_similarity = np.zeros_like(string_similarity)
                self.embed_weight = 0
                self.string_weight = 1.0

        # Weighted combination
        if self.embed_weight > 0:
            combined_similarity = (
                self.string_weight * string_similarity +
                self.embed_weight * embedding_similarity
            )
        else:
            combined_similarity = string_similarity

        return combined_similarity

    def _calculate_string_similarity_matrix(self, titles_a: List[str],
                                          titles_b: List[str]) -> np.ndarray:
        """Calculate string similarity matrix using rapidfuzz."""
        matrix = np.zeros((len(titles_a), len(titles_b)))

        for i, title_a in enumerate(titles_a):
            for j, title_b in enumerate(titles_b):
                similarity = self._calculate_string_similarity(title_a, title_b)
                matrix[i, j] = similarity

        return matrix

    def _calculate_string_similarity(self, text1: str, text2: str) -> float:
        """Calculate string similarity between two texts."""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)

        if text1 == text2:
            return 1.0

        # Use multiple similarity metrics
        ratio = rapidfuzz.fuzz.ratio(text1, text2) / 100.0
        token_ratio = rapidfuzz.fuzz.token_sort_ratio(text1, text2) / 100.0
        partial_ratio = rapidfuzz.fuzz.partial_ratio(text1, text2) / 100.0

        # Return the maximum similarity
        return max(ratio, token_ratio, partial_ratio)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        import re
        # Remove punctuation and normalize whitespace
        text = re.sub(r'[^\w\s\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]', '', text.lower().strip())
        text = re.sub(r'\s+', ' ', text)
        return text

    def _classify_match_type(self, similarity: float) -> str:
        """Classify match type based on similarity score."""
        if similarity >= self.exact_threshold:
            return 'exact'
        elif similarity >= self.partial_threshold:
            return 'partial'
        else:
            return 'poor'

class ParagraphAligner:
    """Aligns paragraphs within matched sections using sequence alignment."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or config.get_align_config()
        self.gap_penalty = self.cfg['gap_penalty']
        self.exact_threshold = self.cfg['exact_threshold']
        self.partial_threshold = self.cfg['partial_threshold']

    def align_paragraphs(self, chunks_a: List[Dict[str, Any]],
                        chunks_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Align paragraphs using sequence alignment algorithm."""
        logger.info(f"Aligning {len(chunks_a)} chunks from A with {len(chunks_b)} chunks from B")

        if not chunks_a and not chunks_b:
            return []

        if not chunks_a:
            return [self._create_alignment(None, chunk_b, 'insert') for chunk_b in chunks_b]

        if not chunks_b:
            return [self._create_alignment(chunk_a, None, 'delete') for chunk_a in chunks_a]

        # Calculate similarity matrix
        similarity_matrix = self._calculate_chunk_similarity_matrix(chunks_a, chunks_b)

        # Perform sequence alignment
        alignment = self._sequence_align(similarity_matrix, chunks_a, chunks_b)

        logger.info(f"Created {len(alignment)} paragraph alignments")
        return alignment

    def _calculate_chunk_similarity_matrix(self, chunks_a: List[Dict[str, Any]],
                                         chunks_b: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate similarity matrix between two sets of chunks."""
        # Prefer body text for comparison if available, fallback to full text
        texts_a = [chunk.get('body_text') or chunk['text'] for chunk in chunks_a]
        texts_b = [chunk.get('body_text') or chunk['text'] for chunk in chunks_b]

        # Get embeddings for semantic similarity
        try:
            embeddings_a = embedding_manager.get_embeddings(texts_a, "chunks_a")
            embeddings_b = embedding_manager.get_embeddings(texts_b, "chunks_b")
            embedding_similarity = embedding_manager.calculate_similarity_matrix(
                embeddings_a, embeddings_b
            )
        except Exception as e:
            logger.warning(f"Failed to compute embeddings: {e}")
            embedding_similarity = np.zeros((len(texts_a), len(texts_b)))

        # Calculate string similarity as fallback
        string_similarity = np.zeros((len(texts_a), len(texts_b)))
        for i, text_a in enumerate(texts_a):
            for j, text_b in enumerate(texts_b):
                string_similarity[i, j] = self._calculate_text_similarity(text_a, text_b)

        # Combine similarities
        if embedding_similarity.size > 0:
            combined = 0.7 * embedding_similarity + 0.3 * string_similarity
        else:
            combined = string_similarity

        return combined

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods."""
        if not text1 or not text2:
            return 0.0

        # Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union

        # Character-level similarity
        char_similarity = rapidfuzz.fuzz.ratio(text1, text2) / 100.0

        # Return weighted average
        return 0.6 * jaccard + 0.4 * char_similarity

    def _sequence_align(self, similarity_matrix: np.ndarray,
                       chunks_a: List[Dict[str, Any]],
                       chunks_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform sequence alignment using dynamic programming (Needleman-Wunsch like)."""
        m, n = len(chunks_a), len(chunks_b)

        # Initialize scoring matrix
        score_matrix = np.zeros((m + 1, n + 1))
        traceback_matrix = np.zeros((m + 1, n + 1), dtype=int)

        # Fill first row and column (gap penalties)
        for i in range(1, m + 1):
            score_matrix[i, 0] = score_matrix[i - 1, 0] - self.gap_penalty
            traceback_matrix[i, 0] = 1  # Up (deletion)

        for j in range(1, n + 1):
            score_matrix[0, j] = score_matrix[0, j - 1] - self.gap_penalty
            traceback_matrix[0, j] = 2  # Left (insertion)

        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Match/mismatch score
                match_score = similarity_matrix[i - 1, j - 1]

                # Three possible operations
                operations = [
                    score_matrix[i - 1, j - 1] + match_score,  # Match/mismatch
                    score_matrix[i - 1, j] - self.gap_penalty,  # Deletion
                    score_matrix[i, j - 1] - self.gap_penalty   # Insertion
                ]

                best_score = max(operations)
                score_matrix[i, j] = best_score

                # Record the operation that gave the best score
                traceback_matrix[i, j] = operations.index(best_score)

        # Traceback to get alignment
        alignment = []
        i, j = m, n

        while i > 0 or j > 0:
            if i == 0:
                # Only insertions left
                alignment.append(self._create_alignment(None, chunks_b[j - 1], 'insert'))
                j -= 1
            elif j == 0:
                # Only deletions left
                alignment.append(self._create_alignment(chunks_a[i - 1], None, 'delete'))
                i -= 1
            else:
                operation = traceback_matrix[i, j]

                if operation == 0:  # Match/mismatch
                    similarity = similarity_matrix[i - 1, j - 1]
                    align_type = self._classify_alignment_type(similarity)
                    alignment.append(self._create_alignment(chunks_a[i - 1], chunks_b[j - 1], align_type))
                    i -= 1
                    j -= 1
                elif operation == 1:  # Deletion
                    alignment.append(self._create_alignment(chunks_a[i - 1], None, 'delete'))
                    i -= 1
                elif operation == 2:  # Insertion
                    alignment.append(self._create_alignment(None, chunks_b[j - 1], 'insert'))
                    j -= 1

        # Reverse to get correct order
        alignment.reverse()
        return alignment

    def _classify_alignment_type(self, similarity: float) -> str:
        """Classify alignment type based on similarity score."""
        if similarity >= self.exact_threshold:
            return 'match'
        elif similarity >= self.partial_threshold:
            return 'partial'
        else:
            return 'replace'

    def _create_alignment(self, chunk_a: Optional[Dict[str, Any]],
                         chunk_b: Optional[Dict[str, Any]],
                         align_type: str) -> Dict[str, Any]:
        """Create alignment object."""
        return {
            'chunk_a': chunk_a,
            'chunk_b': chunk_b,
            'alignment_type': align_type,
            'similarity_score': 0.0 if align_type in ['insert', 'delete'] else
                               self._calculate_alignment_similarity(chunk_a, chunk_b),
            'chunk_a_id': chunk_a['id'] if chunk_a else None,
            'chunk_b_id': chunk_b['id'] if chunk_b else None,
            # Prefer body text for display, fallback to full text
            'chunk_a_text': (chunk_a.get('body_text') or chunk_a['text']) if chunk_a else '',
            'chunk_b_text': (chunk_b.get('body_text') or chunk_b['text']) if chunk_b else '',
            'chunk_a_full_text': chunk_a['text'] if chunk_a else '',  # Keep original for reference
            'chunk_b_full_text': chunk_b['text'] if chunk_b else '',
        }

    def _calculate_alignment_similarity(self, chunk_a: Dict[str, Any],
                                      chunk_b: Dict[str, Any]) -> float:
        """Calculate similarity between two aligned chunks."""
        if not chunk_a or not chunk_b:
            return 0.0

        # Use body text for similarity calculation if available
        text_a = chunk_a.get('body_text') or chunk_a['text']
        text_b = chunk_b.get('body_text') or chunk_b['text']

        return self._calculate_text_similarity(text_a, text_b)

class DocumentAligner:
    """High-level document aligner that coordinates section mapping and paragraph alignment."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or config.get_align_config()
        self.section_mapper = SectionMapper(cfg)
        self.paragraph_aligner = ParagraphAligner(cfg)

    def align_documents(self, doc_a: Dict[str, Any], doc_b: Dict[str, Any]) -> Dict[str, Any]:
        """Align two documents at section and paragraph levels."""
        logger.info("Starting document alignment")
        start_time = time.time()

        sections_a = doc_a.get('sections', [])
        sections_b = doc_b.get('sections', [])

        chunks_a = doc_a.get('chunks', [])
        chunks_b = doc_b.get('chunks', [])

        # Map sections
        section_mappings = self.section_mapper.map_sections(sections_a, sections_b)

        # Align paragraphs within each mapped section
        paragraph_alignments = []
        unmatched_chunks_a = set(chunk['id'] for chunk in chunks_a)
        unmatched_chunks_b = set(chunk['id'] for chunk in chunks_b)

        for mapping in section_mappings:
            if mapping['section_a_data'] and mapping['section_b_data']:
                # Get chunks for this section
                section_chunks_a = [c for c in chunks_a if c['section_id'] == mapping['section_a_id']]
                section_chunks_b = [c for c in chunks_b if c['section_id'] == mapping['section_b_id']]

                # Align paragraphs in this section
                section_alignments = self.paragraph_aligner.align_paragraphs(
                    section_chunks_a, section_chunks_b
                )

                # Add section context to alignments
                for alignment in section_alignments:
                    alignment['section_mapping'] = mapping

                    # Mark chunks as matched
                    if alignment['chunk_a']:
                        unmatched_chunks_a.discard(alignment['chunk_a']['id'])
                    if alignment['chunk_b']:
                        unmatched_chunks_b.discard(alignment['chunk_b']['id'])

                paragraph_alignments.extend(section_alignments)

        # Add unmatched chunks
        for chunk_id in unmatched_chunks_a:
            chunk = next(c for c in chunks_a if c['id'] == chunk_id)
            paragraph_alignments.append(self.paragraph_aligner._create_alignment(
                chunk, None, 'delete'
            ))

        for chunk_id in unmatched_chunks_b:
            chunk = next(c for c in chunks_b if c['id'] == chunk_id)
            paragraph_alignments.append(self.paragraph_aligner._create_alignment(
                None, chunk, 'insert'
            ))

        elapsed = time.time() - start_time
        logger.info(f"Document alignment completed in {elapsed:.2f} seconds")

        return {
            'section_mappings': section_mappings,
            'paragraph_alignments': paragraph_alignments,
            'stats': {
                'sections_a': len(sections_a),
                'sections_b': len(sections_b),
                'chunks_a': len(chunks_a),
                'chunks_b': len(chunks_b),
                'section_mappings': len(section_mappings),
                'paragraph_alignments': len(paragraph_alignments),
                'alignment_time_seconds': elapsed
            }
        }
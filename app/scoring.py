"""Similarity scoring and statistics calculation."""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import json

from .config import config

logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """Calculate document similarity scores and detailed statistics."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or config.get_align_config()

    def calculate_document_similarity(self, alignment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive document similarity metrics."""
        logger.info("Calculating document similarity metrics")

        paragraph_alignments = alignment_result.get('paragraph_alignments', [])
        section_mappings = alignment_result.get('section_mappings', [])

        # Overall similarity
        overall_similarity = self._calculate_overall_similarity(paragraph_alignments)

        # Section-level similarities
        section_similarities = self._calculate_section_similarities(
            section_mappings, paragraph_alignments
        )

        # Detailed statistics
        alignment_stats = self._calculate_alignment_statistics(paragraph_alignments)

        # Content distribution analysis
        content_analysis = self._analyze_content_distribution(paragraph_alignments)

        # Quality metrics
        quality_metrics = self._calculate_quality_metrics(paragraph_alignments)

        result = {
            'overall_similarity': overall_similarity,
            'section_similarities': section_similarities,
            'alignment_statistics': alignment_stats,
            'content_analysis': content_analysis,
            'quality_metrics': quality_metrics,
            'summary': self._create_summary(
                overall_similarity, section_similarities, alignment_stats
            )
        }

        logger.info(f"Document similarity: {overall_similarity:.2f}%")
        return result

    def _calculate_overall_similarity(self, alignments: List[Dict[str, Any]]) -> float:
        """Calculate weighted overall similarity score."""
        if not alignments:
            return 0.0

        total_weight = 0
        weighted_similarity = 0

        for alignment in alignments:
            # Weight by content length (prefer longer matches)
            weight_a = len(alignment.get('chunk_a_text', ''))
            weight_b = len(alignment.get('chunk_b_text', ''))
            weight = max(weight_a, weight_b, 1)  # Minimum weight of 1

            alignment_type = alignment.get('alignment_type', 'unknown')
            similarity = self._get_alignment_similarity_score(alignment, alignment_type)

            weighted_similarity += similarity * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return (weighted_similarity / total_weight) * 100

    def _get_alignment_similarity_score(self, alignment: Dict[str, Any],
                                      alignment_type: str) -> float:
        """Get similarity score for a single alignment."""
        if alignment_type == 'match':
            return 1.0
        elif alignment_type == 'partial':
            return alignment.get('similarity_score', 0.7)
        elif alignment_type == 'replace':
            return alignment.get('similarity_score', 0.3)
        elif alignment_type in ['delete', 'insert']:
            return 0.0
        else:
            return alignment.get('similarity_score', 0.0)

    def _calculate_section_similarities(self, section_mappings: List[Dict[str, Any]],
                                      paragraph_alignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate similarity for each section."""
        section_similarities = []

        # Group alignments by section
        alignments_by_section = defaultdict(list)
        for alignment in paragraph_alignments:
            section_mapping = alignment.get('section_mapping')
            if section_mapping:
                section_key = (
                    section_mapping.get('section_a_id'),
                    section_mapping.get('section_b_id')
                )
                alignments_by_section[section_key].append(alignment)

        # Calculate similarity for each section
        for mapping in section_mappings:
            section_key = (mapping.get('section_a_id'), mapping.get('section_b_id'))
            section_alignments = alignments_by_section.get(section_key, [])

            if section_alignments:
                section_similarity = self._calculate_section_similarity(section_alignments)
            else:
                # Section exists but no paragraph alignments
                section_similarity = 0.0

            section_info = {
                'section_a_id': mapping.get('section_a_id'),
                'section_b_id': mapping.get('section_b_id'),
                'section_a_title': mapping.get('section_a_title', ''),
                'section_b_title': mapping.get('section_b_title', ''),
                'similarity_score': section_similarity,
                'match_type': mapping.get('match_type', 'unknown'),
                'paragraph_count': len(section_alignments),
                'section_level': mapping.get('section_a_level', 1)
            }

            section_similarities.append(section_info)

        # Sort by similarity score (descending)
        section_similarities.sort(key=lambda x: x['similarity_score'], reverse=True)

        return section_similarities

    def _calculate_section_similarity(self, alignments: List[Dict[str, Any]]) -> float:
        """Calculate similarity for a specific section."""
        if not alignments:
            return 0.0

        # Use same logic as overall similarity but for section-specific alignments
        total_weight = 0
        weighted_similarity = 0

        for alignment in alignments:
            weight_a = len(alignment.get('chunk_a_text', ''))
            weight_b = len(alignment.get('chunk_b_text', ''))
            weight = max(weight_a, weight_b, 1)

            alignment_type = alignment.get('alignment_type', 'unknown')
            similarity = self._get_alignment_similarity_score(alignment, alignment_type)

            weighted_similarity += similarity * weight
            total_weight += weight

        return (weighted_similarity / total_weight) * 100 if total_weight > 0 else 0.0

    def _calculate_alignment_statistics(self, alignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed alignment statistics."""
        stats = {
            'total_alignments': len(alignments),
            'alignment_types': Counter(),
            'similarity_distribution': {
                'high': 0,    # > 80%
                'medium': 0,  # 40-80%
                'low': 0      # < 40%
            },
            'content_coverage': {
                'chars_a': 0,
                'chars_b': 0,
                'chars_matched': 0,
                'chars_deleted': 0,
                'chars_inserted': 0,
                'chars_replaced': 0
            }
        }

        similarities = []

        for alignment in alignments:
            alignment_type = alignment.get('alignment_type', 'unknown')
            stats['alignment_types'][alignment_type] += 1

            text_a = alignment.get('chunk_a_text', '')
            text_b = alignment.get('chunk_b_text', '')
            similarity = self._get_alignment_similarity_score(alignment, alignment_type) * 100

            similarities.append(similarity)

            # Similarity distribution
            if similarity > 80:
                stats['similarity_distribution']['high'] += 1
            elif similarity > 40:
                stats['similarity_distribution']['medium'] += 1
            else:
                stats['similarity_distribution']['low'] += 1

            # Content coverage
            stats['content_coverage']['chars_a'] += len(text_a)
            stats['content_coverage']['chars_b'] += len(text_b)

            if alignment_type == 'match':
                stats['content_coverage']['chars_matched'] += max(len(text_a), len(text_b))
            elif alignment_type == 'delete':
                stats['content_coverage']['chars_deleted'] += len(text_a)
            elif alignment_type == 'insert':
                stats['content_coverage']['chars_inserted'] += len(text_b)
            elif alignment_type in ['replace', 'partial']:
                stats['content_coverage']['chars_replaced'] += max(len(text_a), len(text_b))

        # Statistical measures
        if similarities:
            stats['similarity_stats'] = {
                'mean': np.mean(similarities),
                'median': np.median(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities)
            }
        else:
            stats['similarity_stats'] = {
                'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0
            }

        return stats

    def _analyze_content_distribution(self, alignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how content is distributed across alignment types."""
        analysis = {
            'by_type': defaultdict(lambda: {
                'count': 0,
                'total_chars': 0,
                'avg_length': 0,
                'percentage': 0
            }),
            'structure_changes': {
                'reordered_sections': 0,
                'merged_sections': 0,
                'split_sections': 0
            }
        }

        total_chars = 0
        total_count = len(alignments)

        for alignment in alignments:
            alignment_type = alignment.get('alignment_type', 'unknown')
            text_a = alignment.get('chunk_a_text', '')
            text_b = alignment.get('chunk_b_text', '')
            chars = max(len(text_a), len(text_b))

            analysis['by_type'][alignment_type]['count'] += 1
            analysis['by_type'][alignment_type]['total_chars'] += chars
            total_chars += chars

        # Calculate percentages and averages
        for type_name, data in analysis['by_type'].items():
            if data['count'] > 0:
                data['avg_length'] = data['total_chars'] / data['count']
            if total_count > 0:
                data['percentage'] = (data['count'] / total_count) * 100

        return dict(analysis)

    def _calculate_quality_metrics(self, alignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics for the alignment."""
        metrics = {
            'alignment_confidence': 0.0,
            'structure_preservation': 0.0,
            'content_completeness': 0.0,
            'semantic_consistency': 0.0
        }

        if not alignments:
            return metrics

        # Alignment confidence - based on similarity scores
        similarities = [
            self._get_alignment_similarity_score(a, a.get('alignment_type', 'unknown'))
            for a in alignments
        ]
        metrics['alignment_confidence'] = np.mean(similarities) * 100

        # Structure preservation - how well the document structure is maintained
        matched_sections = sum(1 for a in alignments if a.get('alignment_type') == 'match')
        total_sections = len(alignments)
        metrics['structure_preservation'] = (matched_sections / total_sections) * 100 if total_sections > 0 else 0

        # Content completeness - how much content is preserved
        total_content_a = sum(len(a.get('chunk_a_text', '')) for a in alignments)
        total_content_b = sum(len(a.get('chunk_b_text', '')) for a in alignments)
        matched_content = sum(
            max(len(a.get('chunk_a_text', '')), len(a.get('chunk_b_text', '')))
            for a in alignments if a.get('alignment_type') in ['match', 'partial']
        )
        total_content = max(total_content_a, total_content_b)
        metrics['content_completeness'] = (matched_content / total_content) * 100 if total_content > 0 else 0

        # Semantic consistency - placeholder for future semantic analysis
        # This could be enhanced with more sophisticated NLP analysis
        metrics['semantic_consistency'] = metrics['alignment_confidence']

        return metrics

    def _create_summary(self, overall_similarity: float,
                       section_similarities: List[Dict[str, Any]],
                       alignment_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the similarity analysis."""
        # Top and bottom performing sections
        top_sections = section_similarities[:3] if len(section_similarities) >= 3 else section_similarities
        bottom_sections = (section_similarities[-3:] if len(section_similarities) >= 6
                          else section_similarities[3:] if len(section_similarities) > 3 else [])

        # Key statistics
        total_alignments = alignment_stats['total_alignments']
        high_similarity_count = alignment_stats['similarity_distribution']['high']
        match_count = alignment_stats['alignment_types'].get('match', 0)

        summary = {
            'overall_score': overall_similarity,
            'grade': self._get_similarity_grade(overall_similarity),
            'total_sections': len(section_similarities),
            'total_alignments': total_alignments,
            'match_rate': (match_count / total_alignments * 100) if total_alignments > 0 else 0,
            'high_similarity_rate': (high_similarity_count / total_alignments * 100) if total_alignments > 0 else 0,
            'top_performing_sections': [
                {
                    'title': s.get('section_a_title', 'Unknown'),
                    'similarity': s['similarity_score']
                }
                for s in top_sections
            ],
            'improvement_areas': [
                {
                    'title': s.get('section_a_title', 'Unknown'),
                    'similarity': s['similarity_score']
                }
                for s in bottom_sections
            ]
        }

        return summary

    def _get_similarity_grade(self, similarity: float) -> str:
        """Get letter grade for similarity score."""
        if similarity >= 90:
            return 'A'
        elif similarity >= 80:
            return 'B'
        elif similarity >= 70:
            return 'C'
        elif similarity >= 60:
            return 'D'
        else:
            return 'F'

    def export_detailed_scores(self, similarity_result: Dict[str, Any],
                              filepath: str) -> bool:
        """Export detailed scoring results to JSON file."""
        try:
            # Convert numpy types to native Python types for JSON serialization
            serializable_result = self._make_json_serializable(similarity_result)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)

            logger.info(f"Exported detailed scores to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export scores to {filepath}: {e}")
            return False

    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable types to JSON-compatible types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, Counter):
            return dict(obj)
        elif isinstance(obj, defaultdict):
            return dict(obj)
        else:
            return obj
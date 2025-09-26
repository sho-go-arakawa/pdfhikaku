"""Generate visual text differences with color highlighting."""

import re
import difflib
import logging
from typing import List, Dict, Any, Tuple, Optional
from html import escape
import rapidfuzz
from Levenshtein import distance as levenshtein_distance

from .config import config

logger = logging.getLogger(__name__)

class TextDiffer:
    """Generate text differences with advanced highlighting and statistics."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or config.get_align_config()

    def generate_diff(self, alignment: Dict[str, Any],
                     granularity: str = "word") -> Dict[str, Any]:
        """Generate diff for an alignment with HTML highlighting."""
        text_a = alignment.get('chunk_a_text', '')
        text_b = alignment.get('chunk_b_text', '')
        alignment_type = alignment.get('alignment_type', 'unknown')

        if alignment_type == 'delete':
            return self._create_delete_diff(text_a)
        elif alignment_type == 'insert':
            return self._create_insert_diff(text_b)
        elif alignment_type in ['match', 'partial', 'replace']:
            return self._create_comparison_diff(text_a, text_b, granularity)
        else:
            return self._create_empty_diff()

    def _create_delete_diff(self, text: str) -> Dict[str, Any]:
        """Create diff for deleted content."""
        html_text = escape(text)
        return {
            'html_a': f'<span class="delete">{html_text}</span>',
            'html_b': '',
            'stats': {
                'total_tokens': len(text.split()),
                'match_tokens': 0,
                'delete_tokens': len(text.split()),
                'insert_tokens': 0,
                'replace_tokens': 0
            },
            'diff_type': 'delete',
            'similarity': 0.0
        }

    def _create_insert_diff(self, text: str) -> Dict[str, Any]:
        """Create diff for inserted content."""
        html_text = escape(text)
        return {
            'html_a': '',
            'html_b': f'<span class="insert">{html_text}</span>',
            'stats': {
                'total_tokens': len(text.split()),
                'match_tokens': 0,
                'delete_tokens': 0,
                'insert_tokens': len(text.split()),
                'replace_tokens': 0
            },
            'diff_type': 'insert',
            'similarity': 0.0
        }

    def _create_comparison_diff(self, text_a: str, text_b: str,
                              granularity: str) -> Dict[str, Any]:
        """Create comparison diff between two texts."""
        if not text_a and not text_b:
            return self._create_empty_diff()

        if not text_a:
            return self._create_insert_diff(text_b)

        if not text_b:
            return self._create_delete_diff(text_a)

        # Tokenize based on granularity
        tokens_a = self._tokenize(text_a, granularity)
        tokens_b = self._tokenize(text_b, granularity)

        # Generate diff using difflib
        differ = difflib.SequenceMatcher(None, tokens_a, tokens_b)
        opcodes = differ.get_opcodes()

        # Generate HTML and statistics
        html_a, html_b, stats = self._process_opcodes(opcodes, tokens_a, tokens_b)

        # Calculate similarity
        similarity = self._calculate_similarity(stats)

        return {
            'html_a': html_a,
            'html_b': html_b,
            'stats': stats,
            'diff_type': 'comparison',
            'similarity': similarity,
            'opcodes': opcodes
        }

    def _create_empty_diff(self) -> Dict[str, Any]:
        """Create empty diff."""
        return {
            'html_a': '',
            'html_b': '',
            'stats': {
                'total_tokens': 0,
                'match_tokens': 0,
                'delete_tokens': 0,
                'insert_tokens': 0,
                'replace_tokens': 0
            },
            'diff_type': 'empty',
            'similarity': 0.0
        }

    def _tokenize(self, text: str, granularity: str) -> List[str]:
        """Tokenize text based on granularity."""
        if granularity == "character":
            return list(text)
        elif granularity == "word":
            # Enhanced word tokenization that preserves punctuation
            tokens = []
            for match in re.finditer(r'\S+|\s+', text):
                tokens.append(match.group())
            return tokens
        elif granularity == "sentence":
            sentences = self._split_sentences(text)
            return sentences
        elif granularity == "line":
            return text.split('\n')
        else:
            # Default to word
            return text.split()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with Japanese and English support."""
        # Pattern for sentence boundaries
        sentence_pattern = r'(?<=[.!?。！？])\s+(?=[A-Z\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf])'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _process_opcodes(self, opcodes: List[Tuple], tokens_a: List[str],
                        tokens_b: List[str]) -> Tuple[str, str, Dict[str, int]]:
        """Process difflib opcodes to generate HTML and statistics."""
        html_a = []
        html_b = []
        stats = {
            'total_tokens': 0,
            'match_tokens': 0,
            'delete_tokens': 0,
            'insert_tokens': 0,
            'replace_tokens': 0
        }

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                # Matching content
                content_a = ''.join(tokens_a[i1:i2])
                content_b = ''.join(tokens_b[j1:j2])

                html_a.append(f'<span class="match">{escape(content_a)}</span>')
                html_b.append(f'<span class="match">{escape(content_b)}</span>')

                stats['match_tokens'] += (i2 - i1)
                stats['total_tokens'] += (i2 - i1)

            elif tag == 'delete':
                # Deleted content (only in A)
                content = ''.join(tokens_a[i1:i2])
                html_a.append(f'<span class="delete">{escape(content)}</span>')

                stats['delete_tokens'] += (i2 - i1)
                stats['total_tokens'] += (i2 - i1)

            elif tag == 'insert':
                # Inserted content (only in B)
                content = ''.join(tokens_b[j1:j2])
                html_b.append(f'<span class="insert">{escape(content)}</span>')

                stats['insert_tokens'] += (j2 - j1)
                stats['total_tokens'] += (j2 - j1)

            elif tag == 'replace':
                # Replaced content
                content_a = ''.join(tokens_a[i1:i2])
                content_b = ''.join(tokens_b[j1:j2])

                # Check if it's a close match for highlighting
                similarity = self._token_similarity(tokens_a[i1:i2], tokens_b[j1:j2])

                if similarity > 0.6:
                    # Fine-grained diff for similar content
                    fine_html_a, fine_html_b = self._fine_grained_diff(content_a, content_b)
                    html_a.append(f'<span class="partial">{fine_html_a}</span>')
                    html_b.append(f'<span class="partial">{fine_html_b}</span>')
                else:
                    # Complete replacement
                    html_a.append(f'<span class="replace">{escape(content_a)}</span>')
                    html_b.append(f'<span class="replace">{escape(content_b)}</span>')

                stats['replace_tokens'] += max(i2 - i1, j2 - j1)
                stats['total_tokens'] += max(i2 - i1, j2 - j1)

        return ''.join(html_a), ''.join(html_b), stats

    def _token_similarity(self, tokens_a: List[str], tokens_b: List[str]) -> float:
        """Calculate similarity between token sequences."""
        text_a = ''.join(tokens_a)
        text_b = ''.join(tokens_b)

        if not text_a or not text_b:
            return 0.0

        return rapidfuzz.fuzz.ratio(text_a, text_b) / 100.0

    def _fine_grained_diff(self, text_a: str, text_b: str) -> Tuple[str, str]:
        """Generate fine-grained character-level diff for similar texts."""
        # Use character-level diffing for fine details
        chars_a = list(text_a)
        chars_b = list(text_b)

        differ = difflib.SequenceMatcher(None, chars_a, chars_b)
        opcodes = differ.get_opcodes()

        html_a = []
        html_b = []

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                content = ''.join(chars_a[i1:i2])
                html_a.append(escape(content))
                html_b.append(escape(content))
            elif tag == 'delete':
                content = ''.join(chars_a[i1:i2])
                html_a.append(f'<span class="fine-delete">{escape(content)}</span>')
            elif tag == 'insert':
                content = ''.join(chars_b[j1:j2])
                html_b.append(f'<span class="fine-insert">{escape(content)}</span>')
            elif tag == 'replace':
                content_a = ''.join(chars_a[i1:i2])
                content_b = ''.join(chars_b[j1:j2])
                html_a.append(f'<span class="fine-replace">{escape(content_a)}</span>')
                html_b.append(f'<span class="fine-replace">{escape(content_b)}</span>')

        return ''.join(html_a), ''.join(html_b)

    def _calculate_similarity(self, stats: Dict[str, int]) -> float:
        """Calculate similarity percentage from statistics."""
        total = stats['total_tokens']
        if total == 0:
            return 1.0

        matches = stats['match_tokens']
        return (matches / total) * 100.0

class DiffVisualizer:
    """Create visual representations of diffs for UI display."""

    def __init__(self):
        self.differ = TextDiffer()

    def create_side_by_side_diff(self, alignments: List[Dict[str, Any]],
                                granularity: str = "word") -> Dict[str, Any]:
        """Create side-by-side diff visualization."""
        diff_blocks = []
        overall_stats = {
            'total_tokens': 0,
            'match_tokens': 0,
            'delete_tokens': 0,
            'insert_tokens': 0,
            'replace_tokens': 0
        }

        for i, alignment in enumerate(alignments):
            diff = self.differ.generate_diff(alignment, granularity)

            # Accumulate statistics
            for key in overall_stats:
                overall_stats[key] += diff['stats'][key]

            diff_block = {
                'id': f"diff_block_{i}",
                'alignment_type': alignment.get('alignment_type', 'unknown'),
                'section_title': self._get_section_title(alignment),
                'html_a': diff['html_a'],
                'html_b': diff['html_b'],
                'stats': diff['stats'],
                'similarity': diff['similarity'],
                'chunk_a_id': alignment.get('chunk_a_id'),
                'chunk_b_id': alignment.get('chunk_b_id')
            }

            diff_blocks.append(diff_block)

        overall_similarity = self._calculate_overall_similarity(overall_stats)

        return {
            'diff_blocks': diff_blocks,
            'overall_stats': overall_stats,
            'overall_similarity': overall_similarity,
            'total_blocks': len(diff_blocks)
        }

    def _get_section_title(self, alignment: Dict[str, Any]) -> str:
        """Get section title from alignment."""
        section_mapping = alignment.get('section_mapping')
        if section_mapping:
            title_a = section_mapping.get('section_a_title', '')
            title_b = section_mapping.get('section_b_title', '')

            if title_a and title_b:
                return f"{title_a} ↔ {title_b}"
            elif title_a:
                return f"{title_a} (deleted)"
            elif title_b:
                return f"{title_b} (added)"

        return "Content Block"

    def _calculate_overall_similarity(self, stats: Dict[str, int]) -> float:
        """Calculate overall similarity from accumulated statistics."""
        total = stats['total_tokens']
        if total == 0:
            return 100.0

        matches = stats['match_tokens']
        return (matches / total) * 100.0

    def create_unified_diff(self, alignments: List[Dict[str, Any]]) -> str:
        """Create unified diff format (like git diff)."""
        lines = []
        lines.append("--- Document A")
        lines.append("+++ Document B")

        for alignment in alignments:
            alignment_type = alignment.get('alignment_type', 'unknown')
            text_a = alignment.get('chunk_a_text', '')
            text_b = alignment.get('chunk_b_text', '')

            if alignment_type == 'delete':
                for line in text_a.split('\n'):
                    lines.append(f"-{line}")
            elif alignment_type == 'insert':
                for line in text_b.split('\n'):
                    lines.append(f"+{line}")
            elif alignment_type == 'match':
                for line in text_a.split('\n'):
                    lines.append(f" {line}")
            elif alignment_type in ['partial', 'replace']:
                # Show both versions
                for line in text_a.split('\n'):
                    lines.append(f"-{line}")
                for line in text_b.split('\n'):
                    lines.append(f"+{line}")

        return '\n'.join(lines)

    def create_diff_summary(self, alignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics of differences."""
        summary = {
            'total_alignments': len(alignments),
            'matches': 0,
            'partials': 0,
            'replacements': 0,
            'deletions': 0,
            'insertions': 0,
            'sections_compared': set(),
            'avg_similarity': 0.0
        }

        similarities = []

        for alignment in alignments:
            alignment_type = alignment.get('alignment_type', 'unknown')

            if alignment_type == 'match':
                summary['matches'] += 1
                similarities.append(100.0)
            elif alignment_type == 'partial':
                summary['partials'] += 1
                similarities.append(alignment.get('similarity_score', 50.0))
            elif alignment_type == 'replace':
                summary['replacements'] += 1
                similarities.append(alignment.get('similarity_score', 25.0))
            elif alignment_type == 'delete':
                summary['deletions'] += 1
                similarities.append(0.0)
            elif alignment_type == 'insert':
                summary['insertions'] += 1
                similarities.append(0.0)

            # Track sections
            section_mapping = alignment.get('section_mapping')
            if section_mapping:
                if section_mapping.get('section_a_title'):
                    summary['sections_compared'].add(section_mapping['section_a_title'])
                if section_mapping.get('section_b_title'):
                    summary['sections_compared'].add(section_mapping['section_b_title'])

        summary['sections_compared'] = len(summary['sections_compared'])

        if similarities:
            summary['avg_similarity'] = sum(similarities) / len(similarities)

        return summary

def create_diff_css() -> str:
    """Create CSS styles for diff highlighting."""
    return """
    <style>
    .diff-container {
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        line-height: 1.5;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        max-height: 500px;
        overflow-y: auto;
        background: #fafafa;
    }

    .match {
        background-color: white;
        color: #333;
    }

    .delete {
        background-color: #ffebee;
        color: #c62828;
        text-decoration: line-through;
    }

    .insert {
        background-color: #e8f5e8;
        color: #2e7d32;
    }

    .replace {
        background-color: #fff3e0;
        color: #f57c00;
    }

    .partial {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }

    .fine-delete {
        background-color: #ffcdd2;
        color: #d32f2f;
        text-decoration: line-through;
    }

    .fine-insert {
        background-color: #dcedc8;
        color: #388e3c;
    }

    .fine-replace {
        background-color: #ffecb3;
        color: #f57c00;
    }

    .diff-block {
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
    }

    .diff-header {
        background-color: #f5f5f5;
        padding: 8px 12px;
        font-weight: bold;
        border-bottom: 1px solid #e0e0e0;
    }

    .diff-content {
        display: flex;
    }

    .diff-side {
        width: 50%;
        padding: 10px;
    }

    .diff-side:first-child {
        border-right: 1px solid #e0e0e0;
    }

    .diff-stats {
        font-size: 0.85em;
        color: #666;
        margin-top: 5px;
        padding: 5px;
        background-color: #f9f9f9;
        border-radius: 3px;
    }
    </style>
    """
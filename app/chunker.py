"""Hierarchical text chunking based on document structure."""

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from .config import config

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Create hierarchical chunks from PDF pages and headings."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or config.get_chunk_config()
        self.max_tokens = self.cfg['max_tokens']
        self.sentence_split = self.cfg['sentence_split']
        self.overlap_tokens = self.cfg['overlap_tokens']
        self.min_chunk_size = self.cfg['min_chunk_size']

    def create_chunks(self, pages: List[Dict[str, Any]],
                     headings: List[Dict[str, Any]],
                     doc_id: str = "doc") -> List[Dict[str, Any]]:
        """Create hierarchical chunks from pages and headings."""
        logger.info(f"Creating chunks for document {doc_id}")

        # Build hierarchical structure
        sections = self._build_section_hierarchy(pages, headings)

        # Create chunks for each section
        all_chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section, doc_id)
            all_chunks.extend(section_chunks)

        # Post-process chunks
        processed_chunks = self._post_process_chunks(all_chunks)

        logger.info(f"Created {len(processed_chunks)} chunks from {len(sections)} sections")
        return processed_chunks

    def _build_section_hierarchy(self, pages: List[Dict[str, Any]],
                                headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build hierarchical section structure from pages and headings."""
        # Sort headings by page and position
        sorted_headings = sorted(headings, key=lambda h: (h['page'], h.get('line_number', 0)))

        # Create page text lookup (prefer body text if available)
        page_texts = {}
        page_body_texts = {}
        for page in pages:
            page_texts[page['page']] = page['text']
            page_body_texts[page['page']] = page.get('body_text', page['text'])

        sections = []
        current_section = None

        for i, heading in enumerate(sorted_headings):
            # Start new section
            section = {
                'id': f"section_{i + 1}",
                'level': heading['level'],
                'title': heading['text'],
                'page_start': heading['page'],
                'page_end': heading['page'],
                'heading_info': heading,
                'text_parts': [],
                'subsections': []
            }

            # Find end page (start of next section of same or higher level)
            end_page = None
            for j in range(i + 1, len(sorted_headings)):
                next_heading = sorted_headings[j]
                if next_heading['level'] <= heading['level']:
                    end_page = next_heading['page'] - 1
                    break

            if end_page is None:
                end_page = max(page_texts.keys())

            section['page_end'] = max(end_page, heading['page'])

            # Extract text for this section (prefer body text)
            section_text = self._extract_section_text(
                page_texts, heading, section['page_start'], section['page_end']
            )
            section_body_text = self._extract_section_text(
                page_body_texts, heading, section['page_start'], section['page_end']
            )

            section['text_parts'] = [section_text] if section_text else []
            section['body_text_parts'] = [section_body_text] if section_body_text else []
            section['full_text'] = section_text
            section['body_text'] = section_body_text

            sections.append(section)

        # If no headings, create single section from all text
        if not sections:
            all_text = '\n\n'.join(page['text'] for page in pages)
            all_body_text = '\n\n'.join(page.get('body_text', page['text']) for page in pages)
            sections.append({
                'id': 'section_1',
                'level': 1,
                'title': 'Document Content',
                'page_start': 1,
                'page_end': len(pages),
                'heading_info': None,
                'text_parts': [all_text],
                'body_text_parts': [all_body_text],
                'full_text': all_text,
                'body_text': all_body_text,
                'subsections': []
            })

        return sections

    def _extract_section_text(self, page_texts: Dict[int, str],
                             heading: Dict[str, Any],
                             start_page: int, end_page: int) -> str:
        """Extract text for a specific section."""
        section_text_parts = []

        for page_num in range(start_page, end_page + 1):
            if page_num not in page_texts:
                continue

            page_text = page_texts[page_num]

            if page_num == start_page:
                # For first page, try to start after the heading
                text_lines = page_text.split('\n')
                heading_found = False
                page_content = []

                for line in text_lines:
                    if not heading_found:
                        # Look for the heading in the text
                        if self._is_heading_line(line, heading['text']):
                            heading_found = True
                            continue

                    if heading_found or page_num != start_page:
                        page_content.append(line)

                section_text_parts.append('\n'.join(page_content))
            else:
                section_text_parts.append(page_text)

        return '\n\n'.join(section_text_parts).strip()

    def _is_heading_line(self, line: str, heading_text: str) -> bool:
        """Check if a line contains the heading text."""
        line_clean = re.sub(r'[^\w\s]', '', line.lower().strip())
        heading_clean = re.sub(r'[^\w\s]', '', heading_text.lower().strip())

        return heading_clean in line_clean or self._text_similarity(line_clean, heading_clean) > 0.8

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _chunk_section(self, section: Dict[str, Any], doc_id: str) -> List[Dict[str, Any]]:
        """Create chunks from a section."""
        # Prefer body text for chunking, fallback to full text
        text = section.get('body_text') or section['full_text']
        full_text = section['full_text']  # Keep original for reference

        if not text or len(text) < self.min_chunk_size:
            return []

        chunks = []

        if self.sentence_split:
            # Split by sentences first
            sentences = self._split_into_sentences(text)
            chunk_groups = self._group_sentences_into_chunks(sentences)
        else:
            # Split by paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            chunk_groups = self._group_paragraphs_into_chunks(paragraphs)

        # Create chunk objects
        for i, chunk_text in enumerate(chunk_groups):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue

            chunk_id = f"{doc_id}_{section['id']}_chunk_{i + 1}"

            chunks.append({
                'id': chunk_id,
                'doc_id': doc_id,
                'section_id': section['id'],
                'section_path': f"{section['level']}.{section['title'][:50]}",
                'chunk_index': i,
                'text': chunk_text.strip(),
                'body_text': chunk_text.strip(),  # This is already filtered body text
                'full_text_reference': full_text,  # Keep reference to original text
                'token_count': self._estimate_token_count(chunk_text),
                'char_count': len(chunk_text),
                'page_range': (section['page_start'], section['page_end']),
                'section_level': section['level'],
                'section_title': section['title'],
                'meta': {
                    'heading_info': section['heading_info'],
                    'chunk_type': 'sentence' if self.sentence_split else 'paragraph',
                    'section_source': section.get('source', 'unknown'),
                    'toc_entry': section.get('toc_entry'),
                    'is_bookmark_based': section.get('source') == 'bookmark',
                    'uses_body_text': bool(section.get('body_text'))
                }
            })

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with Japanese and English support."""
        # Japanese sentence endings: 。！？
        # English sentence endings: . ! ?
        sentence_pattern = r'[.!?。！？]+(?:\s|$)'

        sentences = re.split(sentence_pattern, text)

        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short sentences
                clean_sentences.append(sentence)

        return clean_sentences

    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """Group sentences into chunks based on token limits."""
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_token_count(sentence)

            # If adding this sentence would exceed limit, finish current chunk
            if (current_token_count + sentence_tokens > self.max_tokens and
                current_chunk):

                chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap if configured
                if self.overlap_tokens > 0:
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk, self.overlap_tokens
                    )
                    current_chunk = overlap_sentences
                    current_token_count = sum(self._estimate_token_count(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_token_count = 0

            current_chunk.append(sentence)
            current_token_count += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _group_paragraphs_into_chunks(self, paragraphs: List[str]) -> List[str]:
        """Group paragraphs into chunks based on token limits."""
        if not paragraphs:
            return []

        chunks = []
        current_chunk = []
        current_token_count = 0

        for paragraph in paragraphs:
            paragraph_tokens = self._estimate_token_count(paragraph)

            # If single paragraph is too large, split it
            if paragraph_tokens > self.max_tokens:
                # Finish current chunk if any
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_token_count = 0

                # Split large paragraph
                para_sentences = self._split_into_sentences(paragraph)
                para_chunks = self._group_sentences_into_chunks(para_sentences)
                chunks.extend(para_chunks)
                continue

            # If adding this paragraph would exceed limit, finish current chunk
            if (current_token_count + paragraph_tokens > self.max_tokens and
                current_chunk):

                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_token_count = 0

            current_chunk.append(paragraph)
            current_token_count += paragraph_tokens

        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """Get sentences for overlap from end of current chunk."""
        overlap_sentences = []
        token_count = 0

        # Take sentences from the end
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_token_count(sentence)
            if token_count + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                token_count += sentence_tokens
            else:
                break

        return overlap_sentences

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 characters per token for mixed languages
        # This is rough but works for chunking purposes
        return max(1, len(text) // 4)

    def _post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process chunks to add additional metadata and validation."""
        processed_chunks = []

        for chunk in chunks:
            # Skip chunks that are too small
            if chunk['char_count'] < self.min_chunk_size:
                continue

            # Add hash for deduplication
            chunk['content_hash'] = hashlib.md5(chunk['text'].encode()).hexdigest()

            # Add summary statistics
            chunk['word_count'] = len(chunk['text'].split())
            chunk['sentence_count'] = len(self._split_into_sentences(chunk['text']))

            # Clean text
            chunk['text'] = self._clean_chunk_text(chunk['text'])

            processed_chunks.append(chunk)

        # Remove near-duplicate chunks
        deduplicated_chunks = self._deduplicate_chunks(processed_chunks)

        return deduplicated_chunks

    def _clean_chunk_text(self, text: str) -> str:
        """Clean chunk text."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)

        # Clean up line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def create_numbered_chapter_chunks(self, pages: List[Dict[str, Any]],
                                     numbered_chapters: List[Dict[str, Any]],
                                     content_start_page: int,
                                     doc_id: str = "doc") -> List[Dict[str, Any]]:
        """Create chunks organized by numbered chapter structure (X-Y.Title format)."""
        logger.info(f"Creating numbered chapter chunks for document {doc_id}")

        if not numbered_chapters:
            logger.warning("No numbered chapters found, falling back to standard chunking")
            return self.create_chunks(pages, numbered_chapters, doc_id)

        # Filter pages to content only (after TOC)
        content_pages = [page for page in pages if page['page'] >= content_start_page]

        # Create page text lookup
        page_texts = {}
        page_body_texts = {}
        for page in content_pages:
            page_texts[page['page']] = page['text']
            page_body_texts[page['page']] = page.get('body_text', page['text'])

        # Organize chapters by numbering hierarchy
        organized_chapters = self._organize_numbered_chapters(numbered_chapters)

        # Create chunks for each chapter
        all_chunks = []
        chapter_chunks = {}

        for chapter_key, chapter_data in organized_chapters.items():
            # Extract text content for this chapter
            chapter_content = self._extract_chapter_content(
                chapter_data, page_texts, page_body_texts, numbered_chapters
            )

            if chapter_content['body_text'].strip():
                # Create chunks for this chapter
                section_chunks = self._create_chapter_chunks(
                    chapter_data, chapter_content, doc_id, chapter_key
                )

                chapter_chunks[chapter_key] = section_chunks
                all_chunks.extend(section_chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(organized_chapters)} numbered chapters")

        return all_chunks

    def _organize_numbered_chapters(self, chapters: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Organize numbered chapters into hierarchical structure."""
        organized = {}

        for chapter in chapters:
            chapter_num = chapter.get('chapter_number', 1)
            section_num = chapter.get('section_number')
            subsection_num = chapter.get('subsection_number')

            # Create hierarchical key
            if subsection_num is not None:
                key = f"{chapter_num}-{section_num}-{subsection_num}"
                level_type = "subsection"
            elif section_num is not None:
                key = f"{chapter_num}-{section_num}"
                level_type = "section"
            else:
                key = f"{chapter_num}"
                level_type = "chapter"

            organized[key] = {
                'chapter_data': chapter,
                'level_type': level_type,
                'chapter_number': chapter_num,
                'section_number': section_num,
                'subsection_number': subsection_num,
                'numbering': chapter['numbering'],
                'title': chapter['title'],
                'start_page': chapter['page']
            }

        return dict(sorted(organized.items(), key=lambda x: self._sort_key_for_numbering(x[0])))

    def _sort_key_for_numbering(self, numbering: str) -> Tuple:
        """Create sort key for numbered chapters."""
        parts = re.findall(r'\d+', numbering)
        # Pad with zeros to ensure proper sorting (1-1 before 1-10)
        return tuple(int(part) for part in parts) + (0,) * (3 - len(parts))

    def _extract_chapter_content(self, chapter_data: Dict,
                               page_texts: Dict[int, str],
                               page_body_texts: Dict[int, str],
                               all_chapters: List[Dict]) -> Dict[str, str]:
        """Extract text content for a specific chapter."""
        start_page = chapter_data['start_page']

        # Find end page (start of next chapter at same or higher level)
        end_page = max(page_texts.keys())  # Default to last page

        for other_chapter in all_chapters:
            other_page = other_chapter['page']
            other_level = other_chapter['level']

            if (other_page > start_page and
                other_level <= chapter_data['chapter_data']['level']):
                end_page = other_page - 1
                break

        # Extract text from pages
        full_text_parts = []
        body_text_parts = []

        for page_num in range(start_page, min(end_page + 1, max(page_texts.keys()) + 1)):
            if page_num in page_texts:
                full_text_parts.append(page_texts[page_num])
                body_text_parts.append(page_body_texts[page_num])

        return {
            'full_text': '\n\n'.join(full_text_parts),
            'body_text': '\n\n'.join(body_text_parts),
            'start_page': start_page,
            'end_page': end_page,
            'page_count': end_page - start_page + 1
        }

    def _create_chapter_chunks(self, chapter_data: Dict, chapter_content: Dict,
                              doc_id: str, chapter_key: str) -> List[Dict[str, Any]]:
        """Create chunks for a single chapter."""
        text = chapter_content['body_text']
        full_text = chapter_content['full_text']

        if not text or len(text) < self.min_chunk_size:
            return []

        chunks = []

        if self.sentence_split:
            # Split by sentences
            sentences = self._split_into_sentences(text)
            chunk_groups = self._group_sentences_into_chunks(sentences)
        else:
            # Split by paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            chunk_groups = self._group_paragraphs_into_chunks(paragraphs)

        # Create chunk objects
        for i, chunk_text in enumerate(chunk_groups):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue

            chunk_id = f"{doc_id}_chapter_{chapter_key}_chunk_{i + 1}"

            chunk = {
                'id': chunk_id,
                'doc_id': doc_id,
                'section_id': f"chapter_{chapter_key}",
                'section_path': f"{chapter_data['level_type']}.{chapter_data['title'][:50]}",
                'chunk_index': i,
                'text': chunk_text.strip(),
                'body_text': chunk_text.strip(),
                'full_text_reference': full_text,
                'token_count': self._estimate_token_count(chunk_text),
                'char_count': len(chunk_text),
                'page_range': (chapter_content['start_page'], chapter_content['end_page']),
                'section_level': chapter_data['chapter_data']['level'],
                'section_title': f"{chapter_data['numbering']} {chapter_data['title']}",
                'chapter_info': {
                    'chapter_number': chapter_data['chapter_number'],
                    'section_number': chapter_data.get('section_number'),
                    'subsection_number': chapter_data.get('subsection_number'),
                    'numbering': chapter_data['numbering'],
                    'level_type': chapter_data['level_type']
                },
                'meta': {
                    'heading_info': chapter_data['chapter_data'],
                    'chunk_type': 'sentence' if self.sentence_split else 'paragraph',
                    'section_source': 'numbered_structure',
                    'is_numbered_chapter': True,
                    'uses_body_text': True
                }
            }

            chunks.append(chunk)

        return chunks

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar chunks."""
        if not chunks:
            return []

        deduplicated = []
        seen_hashes = set()

        for chunk in chunks:
            content_hash = chunk['content_hash']

            if content_hash not in seen_hashes:
                # Check for similar content
                is_similar = False
                for existing in deduplicated:
                    if self._chunks_are_similar(chunk, existing):
                        is_similar = True
                        break

                if not is_similar:
                    deduplicated.append(chunk)
                    seen_hashes.add(content_hash)

        return deduplicated

    def _chunks_are_similar(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> bool:
        """Check if two chunks are very similar."""
        # Check if texts are very similar (>90% overlap)
        text1 = set(chunk1['text'].lower().split())
        text2 = set(chunk2['text'].lower().split())

        if not text1 or not text2:
            return False

        intersection = len(text1 & text2)
        min_length = min(len(text1), len(text2))

        similarity = intersection / min_length if min_length > 0 else 0

        return similarity > 0.9

    def save_chunks_to_file(self, chunks: List[Dict[str, Any]], filepath: str):
        """Save chunks to JSON file for caching."""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(chunks)} chunks to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save chunks to {filepath}: {e}")

    def load_chunks_from_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Load chunks from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"Loaded {len(chunks)} chunks from {filepath}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to load chunks from {filepath}: {e}")
            return []
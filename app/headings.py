"""Enhanced heading detection and table of contents extraction."""

import re
import fitz
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict

from .config import config

logger = logging.getLogger(__name__)

class HeadingDetector:
    """Advanced heading detection using multiple strategies."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or config.get_extract_config()

        # Japanese heading patterns (enhanced for numbered sections)
        self.japanese_patterns = [
            r'^第\d+章\s*(.*)$',        # 第1章
            r'^第\d+節\s*(.*)$',        # 第1節
            r'^第\d+項\s*(.*)$',        # 第1項
            r'^第\d+条\s*(.*)$',        # 第1条
            r'^(\d+)[\.\s](.+)$',       # 1. タイトル (capture number and title)
            r'^(\d+)[-\s](\d+)[\.\s](.+)$',    # 1-1. タイトル (chapter-section format)
            r'^(\d+)\.(\d+)[\.\s](.+)$',       # 1.1. タイトル (dot notation)
            r'^(\d+)\.(\d+)\.(\d+)[\.\s](.+)$', # 1.1.1. タイトル (sub-section)
            r'^[〇一二三四五六七八九十]+[\.\s](.*)$',  # 一. タイトル
        ]

        # English heading patterns
        self.english_patterns = [
            r'^(\d+\.?\d*\.?\d*)\s+(.+)$',  # 1. Title, 1.1 Title
            r'^([A-Z]+\.)\s+(.+)$',         # A. Title
            r'^([IVXLCDM]+\.)\s+(.+)$',     # I. Title
            r'^(Chapter\s+\d+)\s*(.*)$',     # Chapter 1
            r'^(Section\s+\d+)\s*(.*)$',     # Section 1
        ]

        # Special heading words
        self.heading_keywords = [
            # Japanese
            'はじめに', '概要', 'まとめ', '結論', '背景', '目的', '課題', '解決策',
            '提案', '検証', '評価', '考察', '今後の課題', '参考文献', '付録',
            # English
            'introduction', 'overview', 'summary', 'conclusion', 'background',
            'objective', 'methodology', 'results', 'discussion', 'references',
            'appendix', 'abstract'
        ]

    def detect_headings_from_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect headings from extracted pages using multiple strategies."""
        all_headings = []

        # Strategy 1: Font-based detection (if available)
        font_headings = self._detect_font_based_headings(pages)

        # Strategy 2: Pattern-based detection
        pattern_headings = self._detect_pattern_based_headings(pages)

        # Strategy 3: Position-based detection
        position_headings = self._detect_position_based_headings(pages)

        # Combine and deduplicate
        combined_headings = self._combine_and_rank_headings(
            font_headings, pattern_headings, position_headings
        )

        # Post-process and validate
        validated_headings = self._validate_and_clean_headings(combined_headings)

        logger.info(f"Detected {len(validated_headings)} headings across {len(pages)} pages")

        return validated_headings

    def detect_content_start_page(self, pages: List[Dict[str, Any]]) -> int:
        """Detect the page where main content starts (after TOC)."""
        toc_keywords = [
            '目次', 'もくじ', '総目次', '索引', 'contents', 'table of contents',
            'index', '一覧', 'list', '概要', 'overview'
        ]

        content_start_indicators = [
            r'^第?\d+章', r'^\d+[\.\-]\d+', r'^第?\d+節',
            r'はじめに', r'序章', r'introduction', r'序論'
        ]

        toc_end_page = 1

        # Look for TOC pages
        for i, page in enumerate(pages):
            page_text_lower = page['text'].lower()

            # Check if this page contains TOC
            if any(keyword in page_text_lower for keyword in toc_keywords):
                toc_end_page = max(toc_end_page, page['page'])
                continue

            # Look for first substantial content after potential TOC
            lines = page['text'].split('\n')
            content_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]

            if len(content_lines) >= 3:  # Page has substantial content
                # Check if it starts with numbered sections
                for line in content_lines[:5]:  # Check first few lines
                    if any(re.match(pattern, line) for pattern in content_start_indicators):
                        return max(toc_end_page + 1, page['page'])

        # If no clear content start found, assume after first 10% of pages or page 5
        fallback_page = max(5, len(pages) // 10)
        logger.info(f"Content start page detection: Using fallback page {fallback_page}")
        return fallback_page

    def _detect_font_based_headings(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect headings based on font size and formatting."""
        headings = []
        all_font_sizes = []

        # Collect all font sizes for statistical analysis
        for page in pages:
            for span in page.get('spans', []):
                all_font_sizes.append(span['size'])

        if not all_font_sizes:
            return headings

        # Statistical analysis
        font_array = np.array(all_font_sizes)
        mean_size = np.mean(font_array)
        std_size = np.std(font_array)
        q75 = np.percentile(font_array, 75)

        # Dynamic threshold
        heading_threshold = max(mean_size + std_size, q75)

        for page in pages:
            page_headings = self._extract_font_headings_from_page(
                page, heading_threshold, font_array
            )
            headings.extend(page_headings)

        return headings

    def _extract_font_headings_from_page(self, page: Dict[str, Any],
                                       threshold: float, all_sizes: np.ndarray) -> List[Dict[str, Any]]:
        """Extract font-based headings from a single page."""
        headings = []

        # Group spans by line (approximate)
        lines = defaultdict(list)
        for span in page.get('spans', []):
            y_pos = int(span['bbox'][1])  # Top y-coordinate
            lines[y_pos].append(span)

        for y_pos, line_spans in lines.items():
            line_text = ''.join(span['text'] for span in line_spans)
            line_text = line_text.strip()

            if not line_text or len(line_text) > 200:
                continue

            # Get max font size and check for bold
            max_size = max(span['size'] for span in line_spans)
            is_bold = any(span['flags'] & 16 for span in line_spans)  # Bold flag

            # Heading criteria
            is_large_font = max_size >= threshold
            is_short = len(line_text) < 150
            is_not_sentence = not line_text.endswith('。') and not line_text.endswith('.')
            matches_pattern = self._matches_heading_pattern(line_text)

            if (is_large_font or is_bold) and is_short and (is_not_sentence or matches_pattern):
                level = self._calculate_heading_level(max_size, all_sizes, is_bold)

                headings.append({
                    'text': line_text,
                    'level': level,
                    'page': page['page'],
                    'bbox': self._calculate_line_bbox(line_spans),
                    'font_size': max_size,
                    'is_bold': is_bold,
                    'detection_method': 'font',
                    'confidence': self._calculate_font_confidence(max_size, is_bold, matches_pattern)
                })

        return headings

    def _detect_pattern_based_headings(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect headings based on text patterns."""
        headings = []

        for page in pages:
            lines = page['text'].split('\n')

            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line or len(line) > 200:
                    continue

                # Check against all patterns
                heading_info = self._match_heading_patterns(line)
                if heading_info:
                    headings.append({
                        'text': line,
                        'level': heading_info['level'],
                        'page': page['page'],
                        'line_number': line_num + 1,
                        'pattern': heading_info['pattern'],
                        'detection_method': 'pattern',
                        'confidence': heading_info['confidence']
                    })

        return headings

    def _detect_position_based_headings(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect headings based on position and formatting."""
        headings = []

        for page in pages:
            # Look for lines that are:
            # 1. Short
            # 2. Not at the very bottom of page
            # 3. Followed by longer text
            # 4. Match heading keywords

            lines = [line.strip() for line in page['text'].split('\n') if line.strip()]

            for i, line in enumerate(lines):
                if len(line) > 100 or len(line) < 3:
                    continue

                # Check if it's a potential heading
                is_keyword = any(keyword in line.lower() for keyword in self.heading_keywords)
                is_short = len(line) < 80
                has_following_text = i < len(lines) - 1 and len(lines[i + 1]) > 50

                if is_keyword and is_short and has_following_text:
                    headings.append({
                        'text': line,
                        'level': 2,  # Default level for position-based
                        'page': page['page'],
                        'line_number': i + 1,
                        'detection_method': 'position',
                        'confidence': 0.6
                    })

        return headings

    def _match_heading_patterns(self, text: str) -> Optional[Dict[str, Any]]:
        """Match text against heading patterns."""
        # Japanese patterns (higher priority)
        for i, pattern in enumerate(self.japanese_patterns):
            match = re.match(pattern, text)
            if match:
                level = 1 if i < 4 else 2  # Chapter/section patterns get level 1
                return {
                    'level': level,
                    'pattern': pattern,
                    'confidence': 0.9
                }

        # English patterns
        for i, pattern in enumerate(self.english_patterns):
            match = re.match(pattern, text)
            if match:
                level = 1 if 'Chapter' in pattern else 2
                return {
                    'level': level,
                    'pattern': pattern,
                    'confidence': 0.8
                }

        # Special keywords
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in self.heading_keywords):
            return {
                'level': 2,
                'pattern': 'keyword',
                'confidence': 0.7
            }

        return None

    def detect_numbered_chapter_structure(self, pages: List[Dict[str, Any]],
                                         content_start_page: int = 1) -> List[Dict[str, Any]]:
        """Detect chapter structure specifically for X-Y.Title format."""
        chapters = []
        current_chapter = None

        # Enhanced patterns for X-Y.Title format
        chapter_patterns = [
            r'^(\d+)[-\s](\d+)[\.\s]+(.+)$',    # 1-1. タイトル or 1-1 タイトル
            r'^(\d+)\.(\d+)[\.\s]*(.+)$',       # 1.1. タイトル or 1.1 タイトル
            r'^(\d+)\.(\d+)\.(\d+)[\.\s]*(.+)$', # 1.1.1. タイトル
            r'^(\d+)[\.\s]+(.+)$',              # 1. タイトル (main chapter)
        ]

        logger.info(f"Analyzing pages from {content_start_page} onwards for numbered chapters")

        for page in pages:
            if page['page'] < content_start_page:
                continue

            lines = page.get('body_text', page['text']).split('\n')

            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line or len(line) < 5 or len(line) > 150:
                    continue

                chapter_info = self._parse_numbered_heading(line, chapter_patterns)
                if chapter_info:
                    chapter = {
                        'page': page['page'],
                        'line_number': line_num + 1,
                        'text': line,
                        'title': chapter_info['title'],
                        'level': chapter_info['level'],
                        'chapter_number': chapter_info.get('chapter'),
                        'section_number': chapter_info.get('section'),
                        'subsection_number': chapter_info.get('subsection'),
                        'numbering': chapter_info['numbering'],
                        'detection_method': 'numbered_structure',
                        'confidence': chapter_info['confidence']
                    }
                    chapters.append(chapter)

                    if chapter_info['level'] == 1:
                        current_chapter = chapter

        # Post-process to ensure logical ordering
        chapters = self._validate_chapter_sequence(chapters)

        logger.info(f"Detected {len(chapters)} numbered chapters/sections")
        return chapters

    def _parse_numbered_heading(self, text: str, patterns: List[str]) -> Optional[Dict[str, Any]]:
        """Parse numbered heading and extract structure information."""
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()

                if len(groups) == 2:  # Simple numbering: 1. Title
                    chapter = int(groups[0])
                    title = groups[1].strip()
                    return {
                        'chapter': chapter,
                        'title': title,
                        'level': 1,
                        'numbering': f"{chapter}",
                        'confidence': 0.9
                    }

                elif len(groups) == 3:  # Chapter-Section: 1-1. Title
                    chapter = int(groups[0])
                    section = int(groups[1])
                    title = groups[2].strip()
                    return {
                        'chapter': chapter,
                        'section': section,
                        'title': title,
                        'level': 2,
                        'numbering': f"{chapter}-{section}",
                        'confidence': 0.95
                    }

                elif len(groups) == 4:  # Sub-section: 1.1.1. Title
                    chapter = int(groups[0])
                    section = int(groups[1])
                    subsection = int(groups[2])
                    title = groups[3].strip()
                    return {
                        'chapter': chapter,
                        'section': section,
                        'subsection': subsection,
                        'title': title,
                        'level': 3,
                        'numbering': f"{chapter}.{section}.{subsection}",
                        'confidence': 0.9
                    }

        return None

    def _validate_chapter_sequence(self, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean up chapter sequence for logical consistency."""
        if not chapters:
            return chapters

        # Sort by page and line number
        chapters.sort(key=lambda x: (x['page'], x['line_number']))

        validated = []
        seen_numberings = set()

        for chapter in chapters:
            numbering = chapter['numbering']

            # Skip duplicates
            if numbering in seen_numberings:
                continue

            # Basic validation of numbering sequence
            if self._is_valid_chapter_numbering(numbering, validated):
                validated.append(chapter)
                seen_numberings.add(numbering)
            else:
                logger.debug(f"Skipping invalid chapter numbering: {numbering}")

        return validated

    def _is_valid_chapter_numbering(self, numbering: str, existing_chapters: List[Dict]) -> bool:
        """Check if chapter numbering follows logical sequence."""
        if not existing_chapters:
            return True

        # Extract numbers from numbering
        parts = re.findall(r'\d+', numbering)
        if not parts:
            return False

        # For simple validation, just check that numbers are reasonable
        numbers = [int(p) for p in parts]

        # Main chapter should be <= 50, sections <= 20
        if len(numbers) >= 1 and numbers[0] > 50:
            return False
        if len(numbers) >= 2 and numbers[1] > 20:
            return False
        if len(numbers) >= 3 and numbers[2] > 20:
            return False

        return True

    def _matches_heading_pattern(self, text: str) -> bool:
        """Quick check if text matches any heading pattern."""
        return self._match_heading_patterns(text) is not None

    def _calculate_heading_level(self, font_size: float, all_sizes: np.ndarray, is_bold: bool) -> int:
        """Calculate heading level based on font size distribution."""
        unique_sizes = sorted(set(all_sizes), reverse=True)

        # Find size rank
        size_rank = 0
        for i, size in enumerate(unique_sizes):
            if font_size >= size:
                size_rank = i
                break

        # Adjust for bold
        if is_bold and size_rank > 0:
            size_rank -= 1

        return min(max(size_rank + 1, 1), 6)

    def _calculate_line_bbox(self, spans: List[Dict[str, Any]]) -> List[float]:
        """Calculate bounding box for a line of spans."""
        if not spans:
            return [0, 0, 0, 0]

        min_x = min(span['bbox'][0] for span in spans)
        min_y = min(span['bbox'][1] for span in spans)
        max_x = max(span['bbox'][2] for span in spans)
        max_y = max(span['bbox'][3] for span in spans)

        return [min_x, min_y, max_x, max_y]

    def _calculate_font_confidence(self, font_size: float, is_bold: bool, matches_pattern: bool) -> float:
        """Calculate confidence score for font-based detection."""
        confidence = 0.5

        if font_size > 14:
            confidence += 0.2
        if is_bold:
            confidence += 0.2
        if matches_pattern:
            confidence += 0.3

        return min(confidence, 1.0)

    def _combine_and_rank_headings(self, *heading_lists) -> List[Dict[str, Any]]:
        """Combine headings from different methods and rank by confidence."""
        all_headings = []
        for heading_list in heading_lists:
            all_headings.extend(heading_list)

        # Remove duplicates based on text similarity and page
        deduplicated = self._remove_duplicate_headings(all_headings)

        # Sort by page and confidence
        deduplicated.sort(key=lambda x: (x['page'], -x.get('confidence', 0.5)))

        return deduplicated

    def _remove_duplicate_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate headings based on text similarity."""
        if not headings:
            return []

        deduplicated = []

        for heading in headings:
            is_duplicate = False

            for existing in deduplicated:
                # Same page and very similar text
                if (existing['page'] == heading['page'] and
                    self._text_similarity(existing['text'], heading['text']) > 0.8):

                    # Keep the one with higher confidence
                    if heading.get('confidence', 0.5) > existing.get('confidence', 0.5):
                        deduplicated.remove(existing)
                        deduplicated.append(heading)

                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(heading)

        return deduplicated

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple character-based metric."""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        text1 = re.sub(r'[^\w]', '', text1.lower())
        text2 = re.sub(r'[^\w]', '', text2.lower())

        if text1 == text2:
            return 1.0

        # Calculate Jaccard similarity
        set1 = set(text1)
        set2 = set(text2)

        if not set1 and not set2:
            return 1.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _validate_and_clean_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean detected headings."""
        validated = []

        for heading in headings:
            text = heading['text'].strip()

            # Skip if too short or too long
            if len(text) < 2 or len(text) > 200:
                continue

            # Skip if looks like regular text (ends with period and is long)
            if len(text) > 50 and (text.endswith('。') or text.endswith('.')):
                continue

            # Clean the text
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            heading['text'] = text

            # Ensure minimum confidence
            if heading.get('confidence', 0.5) >= 0.4:
                validated.append(heading)

        return validated

class TOCExtractor:
    """Extract table of contents from PDF bookmarks or dedicated TOC pages."""

    def __init__(self):
        pass

    def extract_comprehensive_toc(self, pdf_path: str, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract comprehensive TOC prioritizing bookmarks over text detection."""
        # Primary: Extract from bookmarks (most reliable)
        toc_bookmarks = self.extract_toc_from_bookmarks(pdf_path)

        if toc_bookmarks:
            logger.info(f"Using PDF bookmarks as primary TOC source ({len(toc_bookmarks)} entries)")

            # Enhance bookmark TOC with sections from text headers
            enhanced_toc = self._enhance_bookmark_toc_with_sections(toc_bookmarks, pages)
            return enhanced_toc

        # Fallback: Extract from text if no bookmarks available
        logger.info("No PDF bookmarks found, falling back to text-based TOC extraction")
        toc_text = self.extract_toc_from_text(pages)

        if toc_text:
            logger.info(f"Using text-based TOC ({len(toc_text)} entries)")
            # Enhance text-based TOC with section detection
            enhanced_text_toc = self._enhance_text_toc_with_sections(toc_text, pages)
            return enhanced_text_toc

        # Final fallback: Create minimal structure
        logger.warning("No TOC found, will rely on heading detection")
        return []

    def _enhance_bookmark_toc(self, bookmarks: List[Dict[str, Any]], pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance bookmark-based TOC with additional structure information."""
        enhanced_toc = []

        for bookmark in bookmarks:
            enhanced_entry = bookmark.copy()

            # Add additional metadata
            enhanced_entry['confidence'] = 1.0  # Bookmarks are highly reliable
            enhanced_entry['detection_method'] = 'bookmark_enhanced'

            # Try to find the actual heading text on the page for better matching
            page_text = self._get_page_text(pages, bookmark['page'])
            if page_text:
                # Look for heading text that matches bookmark title
                heading_match = self._find_heading_in_text(page_text, bookmark['title'])
                if heading_match:
                    enhanced_entry['heading_text'] = heading_match
                    enhanced_entry['text_match_confidence'] = 0.9

            enhanced_toc.append(enhanced_entry)

        return enhanced_toc

    def _get_page_text(self, pages: List[Dict[str, Any]], page_num: int) -> str:
        """Get text content for a specific page."""
        for page in pages:
            if page['page'] == page_num:
                return page['text']
        return ""

    def _find_heading_in_text(self, page_text: str, bookmark_title: str) -> Optional[str]:
        """Find heading text in page that matches bookmark title."""
        lines = page_text.split('\n')
        bookmark_clean = re.sub(r'[^\w\s]', '', bookmark_title.lower().strip())

        for line in lines:
            line_clean = re.sub(r'[^\w\s]', '', line.lower().strip())
            if bookmark_clean in line_clean or self._text_similarity(bookmark_clean, line_clean) > 0.7:
                return line.strip()

        return None

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

    def _enhance_bookmark_toc_with_sections(self, toc_bookmarks: List[Dict[str, Any]], pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance bookmark TOC with sections extracted from text headers (X-Y.Title format)."""
        enhanced_toc = toc_bookmarks.copy()

        # Extract sections from text headers
        text_sections = self._extract_section_headers_from_text(pages)

        if text_sections:
            logger.info(f"Found {len(text_sections)} section headers in text")

            # Organize sections by chapter
            chapters_with_sections = self._organize_sections_by_chapters(toc_bookmarks, text_sections)

            # Rebuild the TOC with integrated sections
            enhanced_toc = self._integrate_sections_into_toc(toc_bookmarks, chapters_with_sections)

        return enhanced_toc

    def _enhance_text_toc_with_sections(self, toc_text: List[Dict[str, Any]], pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance text-based TOC with sections extracted from text headers."""
        enhanced_toc = toc_text.copy()

        # Extract sections from text headers
        text_sections = self._extract_section_headers_from_text(pages)

        if text_sections:
            logger.info(f"Found {len(text_sections)} section headers in text")

            # Organize sections by chapter
            chapters_with_sections = self._organize_sections_by_chapters(toc_text, text_sections)

            # Rebuild the TOC with integrated sections
            enhanced_toc = self._integrate_sections_into_toc(toc_text, chapters_with_sections)

        return enhanced_toc

    def _extract_section_headers_from_text(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract section headers in X-Y.Title format from page text."""
        sections = []

        # Patterns for section headers (X-Y.Title format)
        section_patterns = [
            r'^(\d+)[-\s](\d+)[\.\s]*(.+)$',      # 2-3. タイトル, 2-3 タイトル
            r'^(\d+)\.(\d+)[\.\s]*(.+)$',         # 2.3. タイトル, 2.3 タイトル
            r'^(\d+)[-\s](\d+)[-\s](\d+)[\.\s]*(.+)$',  # 2-3-1. タイトル (sub-section)
            r'^(\d+)\.(\d+)\.(\d+)[\.\s]*(.+)$',        # 2.3.1. タイトル (sub-section)
        ]

        for page in pages:
            lines = page.get('body_text', page['text']).split('\n')

            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line or len(line) < 5 or len(line) > 150:
                    continue

                for pattern in section_patterns:
                    match = re.match(pattern, line)
                    if match:
                        groups = match.groups()

                        if len(groups) == 3:  # X-Y.Title format
                            chapter_num = int(groups[0])
                            section_num = int(groups[1])
                            title = groups[2].strip()

                            sections.append({
                                'chapter_number': chapter_num,
                                'section_number': section_num,
                                'title': title,
                                'full_text': line,
                                'page': page['page'],
                                'line_number': line_num + 1,
                                'level': 2,  # Section level
                                'source': 'text_header',
                                'confidence': 0.9
                            })
                            break

                        elif len(groups) == 4:  # X-Y-Z.Title format (sub-section)
                            chapter_num = int(groups[0])
                            section_num = int(groups[1])
                            subsection_num = int(groups[2])
                            title = groups[3].strip()

                            sections.append({
                                'chapter_number': chapter_num,
                                'section_number': section_num,
                                'subsection_number': subsection_num,
                                'title': title,
                                'full_text': line,
                                'page': page['page'],
                                'line_number': line_num + 1,
                                'level': 3,  # Sub-section level
                                'source': 'text_header',
                                'confidence': 0.85
                            })
                            break

        # Sort sections by chapter, section, and subsection numbers
        sections.sort(key=lambda x: (
            x['chapter_number'],
            x['section_number'],
            x.get('subsection_number', 0)
        ))

        return sections

    def _organize_sections_by_chapters(self, toc_chapters: List[Dict[str, Any]], text_sections: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Organize sections by their chapter numbers."""
        chapters_with_sections = {}

        # Initialize with existing chapter numbers from TOC
        for toc_entry in toc_chapters:
            # Try to extract chapter number from TOC title
            chapter_num = self._extract_chapter_number_from_title(toc_entry.get('title', ''))
            if chapter_num:
                if chapter_num not in chapters_with_sections:
                    chapters_with_sections[chapter_num] = []

        # Group sections by chapter
        for section in text_sections:
            chapter_num = section['chapter_number']
            if chapter_num not in chapters_with_sections:
                chapters_with_sections[chapter_num] = []
            chapters_with_sections[chapter_num].append(section)

        return chapters_with_sections

    def _extract_chapter_number_from_title(self, title: str) -> Optional[int]:
        """Extract chapter number from TOC title."""
        if not title:
            return None

        # Patterns for chapter identification
        chapter_patterns = [
            r'^第?(\d+)章',       # 第2章, 2章
            r'^(\d+)[\.\s]',      # 2. タイトル, 2 タイトル
            r'^Chapter\s*(\d+)',  # Chapter 2
        ]

        for pattern in chapter_patterns:
            match = re.match(pattern, title.strip())
            if match:
                return int(match.group(1))

        return None

    def _integrate_sections_into_toc(self, original_toc: List[Dict[str, Any]], chapters_with_sections: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Integrate sections into the original TOC structure."""
        enhanced_toc = []

        for toc_entry in original_toc:
            # Add the original chapter entry
            enhanced_toc.append(toc_entry)

            # Check if this chapter has sections
            chapter_num = self._extract_chapter_number_from_title(toc_entry.get('title', ''))
            if chapter_num and chapter_num in chapters_with_sections:
                sections = chapters_with_sections[chapter_num]

                # Add sections under this chapter
                for section in sections:
                    section_entry = {
                        'id': f"section_{chapter_num}_{section['section_number']}",
                        'level': section['level'],
                        'title': f"{chapter_num}-{section['section_number']}. {section['title']}",
                        'page': section['page'],
                        'source': 'text_header',
                        'confidence': section['confidence'],
                        'parent_chapter': chapter_num,
                        'section_info': section
                    }

                    enhanced_toc.append(section_entry)

                    # Add sub-sections if any
                    if section.get('subsection_number'):
                        # This is already a sub-section, add it appropriately
                        pass

        return enhanced_toc

    def extract_toc_from_bookmarks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract TOC from PDF bookmarks with enhanced metadata."""
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            doc.close()

            if not toc:
                return []

            outline = []
            for level, title, page in toc:
                # Clean and validate bookmark entry
                clean_title = title.strip()
                if not clean_title or len(clean_title) > 200:
                    continue

                outline.append({
                    "id": f"bookmark_{len(outline) + 1}",
                    "level": max(1, level),  # Ensure level is at least 1
                    "title": clean_title,
                    "page": max(1, page),  # Ensure page is at least 1
                    "source": "bookmark",
                    "confidence": 1.0,
                    "detection_method": "pdf_bookmark"
                })

            logger.info(f"Extracted {len(outline)} items from PDF bookmarks")
            return self._validate_and_sort_bookmarks(outline)

        except Exception as e:
            logger.warning(f"Failed to extract TOC from bookmarks: {e}")
            return []

    def _validate_and_sort_bookmarks(self, bookmarks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and sort bookmark entries."""
        if not bookmarks:
            return []

        # Sort by page number first, then by level
        sorted_bookmarks = sorted(bookmarks, key=lambda x: (x['page'], x['level']))

        # Validate hierarchy (ensure level progression is logical)
        validated_bookmarks = []
        previous_level = 0

        for bookmark in sorted_bookmarks:
            current_level = bookmark['level']

            # Adjust level if it jumps too much from previous
            if validated_bookmarks:
                max_allowed_level = previous_level + 2
                if current_level > max_allowed_level:
                    current_level = max_allowed_level
                    bookmark['level'] = current_level
                    bookmark['level_adjusted'] = True

            validated_bookmarks.append(bookmark)
            previous_level = current_level

        return validated_bookmarks

    def extract_toc_from_text(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract TOC by finding dedicated table of contents pages."""
        toc_pages = self._find_toc_pages(pages)

        if not toc_pages:
            return []

        toc_entries = []
        for page in toc_pages:
            entries = self._parse_toc_page(page)
            toc_entries.extend(entries)

        logger.info(f"Extracted {len(toc_entries)} TOC entries from text")
        return toc_entries

    def _find_toc_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find pages that contain table of contents."""
        toc_pages = []

        toc_keywords = [
            '目次', 'contents', 'table of contents', '索引', 'index'
        ]

        for page in pages[:10]:  # Usually TOC is in first few pages
            text_lower = page['text'].lower()

            # Check for TOC keywords
            has_toc_keyword = any(keyword in text_lower for keyword in toc_keywords)

            # Check for TOC-like patterns (lines with page numbers)
            toc_pattern_count = len(re.findall(r'.+\.\.\.\s*\d+', page['text']))
            toc_pattern_count += len(re.findall(r'.+\s+\d+$', page['text'], re.MULTILINE))

            if has_toc_keyword or toc_pattern_count > 5:
                toc_pages.append(page)

        return toc_pages

    def _parse_toc_page(self, page: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse a table of contents page."""
        entries = []
        lines = page['text'].split('\n')

        patterns = [
            r'^(.+?)\.{3,}\s*(\d+)$',  # Title...123
            r'^(.+?)\s+(\d+)$',        # Title 123
            r'^(\d+\.?\d*)\s+(.+?)\s+(\d+)$',  # 1.1 Title 123
        ]

        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue

            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    if len(match.groups()) == 2:
                        title, page_num = match.groups()
                        level = 1
                    else:
                        number, title, page_num = match.groups()
                        level = number.count('.') + 1

                    try:
                        page_num = int(page_num)
                        entries.append({
                            "id": f"toc_{len(entries) + 1}",
                            "level": level,
                            "title": title.strip(),
                            "page": page_num,
                            "source": "toc_text"
                        })
                    except ValueError:
                        continue
                    break

        return entries
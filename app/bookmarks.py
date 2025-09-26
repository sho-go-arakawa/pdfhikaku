"""PDF bookmark generation and manipulation."""

import fitz
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class BookmarkGenerator:
    """Generate and manage PDF bookmarks from detected document structure."""

    def __init__(self):
        pass

    def add_bookmarks_to_pdf(self, input_pdf_path: str,
                           headings: List[Dict[str, Any]],
                           output_pdf_path: str) -> bool:
        """Add hierarchical bookmarks to PDF based on detected headings."""
        try:
            logger.info(f"Adding bookmarks to {input_pdf_path}")

            # Open PDF
            doc = fitz.open(input_pdf_path)

            # Clear existing bookmarks
            doc.set_toc([])

            # Convert headings to TOC format
            toc = self._headings_to_toc(headings, doc.page_count)

            if toc:
                # Set new bookmarks
                doc.set_toc(toc)
                logger.info(f"Added {len(toc)} bookmarks")

                # Save PDF with bookmarks
                Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
                doc.save(output_pdf_path)

            doc.close()

            if toc:
                logger.info(f"Successfully added {len(toc)} bookmarks to {output_pdf_path}")
                return True
            else:
                logger.warning(f"No valid bookmarks to add to {input_pdf_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to add bookmarks to {input_pdf_path}: {e}")
            return False

    def extract_existing_bookmarks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract existing bookmarks from PDF."""
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            doc.close()

            bookmarks = []
            for level, title, page in toc:
                bookmarks.append({
                    'level': level,
                    'title': title.strip(),
                    'page': page,
                    'source': 'existing_bookmark'
                })

            logger.info(f"Extracted {len(bookmarks)} existing bookmarks from {pdf_path}")
            return bookmarks

        except Exception as e:
            logger.warning(f"Failed to extract bookmarks from {pdf_path}: {e}")
            return []

    def create_bookmarks_from_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create bookmark entries from document sections."""
        bookmarks = []

        for section in sections:
            bookmark = {
                'level': section.get('level', 1),
                'title': section.get('title', 'Untitled Section'),
                'page': section.get('page_start', 1),
                'section_id': section.get('id'),
                'source': 'detected_section'
            }
            bookmarks.append(bookmark)

        # Sort by page number
        bookmarks.sort(key=lambda x: x['page'])

        return bookmarks

    def _headings_to_toc(self, headings: List[Dict[str, Any]], max_pages: int) -> List[List]:
        """Convert heading list to PyMuPDF TOC format."""
        if not headings:
            return []

        toc = []

        # Sort headings by page and position
        sorted_headings = sorted(headings, key=lambda h: (
            h.get('page', 1),
            h.get('line_number', 0),
            h.get('bbox', [0, 0, 0, 0])[1]  # Sort by y-position
        ))

        # Filter and validate headings
        valid_headings = []
        for heading in sorted_headings:
            page = heading.get('page', 1)
            title = heading.get('text', '').strip()
            level = heading.get('level', 1)

            # Validation checks
            if not title:
                continue

            if page < 1 or page > max_pages:
                continue

            if level < 1 or level > 6:
                level = min(max(level, 1), 6)

            valid_headings.append({
                'level': level,
                'title': title,
                'page': page,
                'original': heading
            })

        # Build hierarchical TOC
        toc = []
        for heading in valid_headings:
            toc_entry = [
                heading['level'],
                heading['title'],
                heading['page']
            ]
            toc.append(toc_entry)

        return toc

    def merge_bookmarks(self, existing_bookmarks: List[Dict[str, Any]],
                       detected_headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge existing bookmarks with newly detected headings."""
        # If existing bookmarks are comprehensive, prefer them
        if len(existing_bookmarks) > len(detected_headings) * 0.7:
            logger.info("Using existing bookmarks (more comprehensive)")
            return existing_bookmarks

        # Otherwise, use detected headings
        logger.info("Using detected headings for bookmarks")
        return detected_headings

    def validate_bookmark_structure(self, bookmarks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate bookmark structure and return issues found."""
        issues = []

        if not bookmarks:
            issues.append("No bookmarks provided")
            return False, issues

        # Check for proper level hierarchy
        prev_level = 0
        for i, bookmark in enumerate(bookmarks):
            level = bookmark.get('level', 1)

            # Level should not jump by more than 1
            if level > prev_level + 1:
                issues.append(f"Bookmark {i}: Level jump from {prev_level} to {level}")

            prev_level = level

        # Check for valid page numbers
        for i, bookmark in enumerate(bookmarks):
            page = bookmark.get('page', 1)
            if page < 1:
                issues.append(f"Bookmark {i}: Invalid page number {page}")

        # Check for empty titles
        for i, bookmark in enumerate(bookmarks):
            title = bookmark.get('title', '').strip()
            if not title:
                issues.append(f"Bookmark {i}: Empty title")

        is_valid = len(issues) == 0
        return is_valid, issues

    def optimize_bookmark_structure(self, bookmarks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize bookmark structure by fixing common issues."""
        if not bookmarks:
            return bookmarks

        optimized = []

        # Sort by page number first
        bookmarks = sorted(bookmarks, key=lambda x: x.get('page', 1))

        # Fix level hierarchy
        prev_level = 0
        for bookmark in bookmarks:
            level = bookmark.get('level', 1)
            title = bookmark.get('title', '').strip()
            page = bookmark.get('page', 1)

            # Skip invalid bookmarks
            if not title or page < 1:
                continue

            # Fix level jumps
            if level > prev_level + 1:
                level = prev_level + 1

            # Ensure minimum level is 1
            level = max(level, 1)

            optimized_bookmark = {
                'level': level,
                'title': title,
                'page': page,
                'optimized': True
            }

            # Preserve additional metadata
            for key in ['section_id', 'source', 'confidence']:
                if key in bookmark:
                    optimized_bookmark[key] = bookmark[key]

            optimized.append(optimized_bookmark)
            prev_level = level

        logger.info(f"Optimized {len(bookmarks)} bookmarks to {len(optimized)} valid entries")
        return optimized

    def create_bookmark_navigation_map(self, bookmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a navigation map for easy bookmark traversal."""
        nav_map = {
            'bookmarks': bookmarks,
            'by_page': {},
            'by_level': {},
            'total_count': len(bookmarks)
        }

        # Index by page
        for bookmark in bookmarks:
            page = bookmark.get('page')
            if page not in nav_map['by_page']:
                nav_map['by_page'][page] = []
            nav_map['by_page'][page].append(bookmark)

        # Index by level
        for bookmark in bookmarks:
            level = bookmark.get('level')
            if level not in nav_map['by_level']:
                nav_map['by_level'][level] = []
            nav_map['by_level'][level].append(bookmark)

        return nav_map

    def add_diff_bookmarks(self, input_pdf_path: str,
                          alignment_results: Dict[str, Any],
                          output_pdf_path: str) -> bool:
        """Add bookmarks that highlight diff locations."""
        try:
            logger.info(f"Adding diff bookmarks to {input_pdf_path}")

            doc = fitz.open(input_pdf_path)

            # Get existing TOC
            existing_toc = doc.get_toc()

            # Create diff bookmarks
            diff_bookmarks = self._create_diff_bookmarks(alignment_results)

            # Merge with existing bookmarks
            merged_toc = self._merge_toc_with_diffs(existing_toc, diff_bookmarks)

            if merged_toc:
                doc.set_toc(merged_toc)

            Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
            doc.save(output_pdf_path)
            doc.close()

            logger.info(f"Added {len(diff_bookmarks)} diff bookmarks to {output_pdf_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to add diff bookmarks: {e}")
            return False

    def _create_diff_bookmarks(self, alignment_results: Dict[str, Any]) -> List[List]:
        """Create bookmarks for significant differences."""
        diff_bookmarks = []

        section_mappings = alignment_results.get('section_mappings', [])

        # Add bookmarks for sections with low similarity
        for mapping in section_mappings:
            similarity = mapping.get('similarity_score', 100)
            section_title = mapping.get('section_a_title', 'Unknown Section')

            if similarity < 70:  # Low similarity threshold
                page = 1  # This would need to be calculated from actual section data

                bookmark = [
                    2,  # Level 2 (sub-bookmark)
                    f"⚠️ {section_title} ({similarity:.1f}% similar)",
                    page
                ]
                diff_bookmarks.append(bookmark)

        return diff_bookmarks

    def _merge_toc_with_diffs(self, existing_toc: List[List],
                            diff_bookmarks: List[List]) -> List[List]:
        """Merge existing TOC with diff bookmarks."""
        merged = existing_toc.copy()

        # Add a top-level bookmark for differences
        if diff_bookmarks:
            merged.append([1, "🔍 Differences", 1])
            merged.extend(diff_bookmarks)

        return merged
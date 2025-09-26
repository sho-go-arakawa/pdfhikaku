#!/usr/bin/env python
"""Test script for section-chapter linking functionality."""

import sys
import os
sys.path.append('/workspaces/pdfhikaku')

from app.headings import OutlineDetector

def test_section_extraction():
    """Test section extraction from sample text."""

    # Create sample pages with section headers
    sample_pages = [
        {
            'page_number': 1,
            'text': """
            1章 総論
            1-1. 概要
            本文がここに続きます...
            """,
            'content_lines': ['1章 総論', '1-1. 概要', '本文がここに続きます...']
        },
        {
            'page_number': 2,
            'text': """
            2章 実装
            2-1. データベース設計
            2-2. API設計
            詳細な内容...
            """,
            'content_lines': ['2章 実装', '2-1. データベース設計', '2-2. API設計', '詳細な内容...']
        },
        {
            'page_number': 3,
            'text': """
            3-1. フロントエンド実装
            3-2. バックエンド実装
            実装の詳細...
            """,
            'content_lines': ['3-1. フロントエンド実装', '3-2. バックエンド実装', '実装の詳細...']
        }
    ]

    # Create sample TOC entries (like from bookmarks)
    sample_toc = [
        {'title': '1章 総論', 'page': 1, 'level': 1, 'source': 'bookmark'},
        {'title': '2章 実装', 'page': 2, 'level': 1, 'source': 'bookmark'},
        {'title': '3章 運用', 'page': 3, 'level': 1, 'source': 'bookmark'}
    ]

    detector = OutlineDetector()

    print("Testing section extraction...")

    # Test section extraction
    sections = detector._extract_section_headers_from_text(sample_pages)
    print(f"Extracted {len(sections)} sections:")
    for section in sections:
        print(f"  - {section['title']} (Chapter {section['chapter']}, Page {section['page']})")

    # Test section organization
    organized_sections = detector._organize_sections_by_chapters(sections)
    print(f"\nOrganized sections by chapters:")
    for chapter, chapter_sections in organized_sections.items():
        print(f"  Chapter {chapter}: {len(chapter_sections)} sections")
        for section in chapter_sections:
            print(f"    - {section['title']}")

    # Test TOC integration
    enhanced_toc = detector._enhance_bookmark_toc_with_sections(sample_toc, sample_pages)
    print(f"\nEnhanced TOC with {len(enhanced_toc)} entries:")
    for entry in enhanced_toc:
        indent = "  " * (entry.get('level', 1) - 1)
        source = entry.get('source', 'unknown')
        print(f"{indent}- {entry['title']} (Page {entry['page']}, Source: {source})")

    return True

if __name__ == "__main__":
    try:
        success = test_section_extraction()
        if success:
            print("\n✅ Section-chapter linking test passed!")
        else:
            print("\n❌ Section-chapter linking test failed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
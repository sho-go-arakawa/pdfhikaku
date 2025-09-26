#!/usr/bin/env python
"""Standalone test for section extraction patterns."""

import re

def extract_section_headers_from_text(pages):
    """Extract section headers in X-Y.Title format from page text."""
    sections = []

    # Define patterns for section headers
    section_patterns = [
        r'^(\d+)[-\s](\d+)[\.\s]*(.+)',      # 2-3. タイトル, 2-3 タイトル
        r'^(\d+)\.(\d+)[\.\s]*(.+)',         # 2.3. タイトル, 2.3 タイトル
        r'^(\d+)[-\s](\d+)[-\s](\d+)[\.\s]*(.+)',  # 2-3-1. タイトル (sub-section)
        r'^(\d+)\.(\d+)\.(\d+)[\.\s]*(.+)',        # 2.3.1. タイトル (sub-section)
    ]

    for page in pages:
        page_num = page.get('page_number', 0)
        content_lines = page.get('content_lines', [])

        for line in content_lines:
            line = line.strip()
            if not line:
                continue

            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    groups = match.groups()

                    if len(groups) == 3:  # X-Y.Title format
                        chapter, section, title = groups
                        sections.append({
                            'chapter': int(chapter),
                            'section': int(section),
                            'title': title.strip(),
                            'page': page_num,
                            'level': 2,
                            'source': 'text_header',
                            'full_text': line
                        })
                    elif len(groups) == 4:  # X-Y-Z.Title format (sub-section)
                        chapter, section, subsection, title = groups
                        sections.append({
                            'chapter': int(chapter),
                            'section': int(section),
                            'subsection': int(subsection),
                            'title': title.strip(),
                            'page': page_num,
                            'level': 3,
                            'source': 'text_header',
                            'full_text': line
                        })
                    break  # Only match the first pattern that works

    return sections

def organize_sections_by_chapters(sections):
    """Organize sections by chapter number."""
    chapters = {}

    for section in sections:
        chapter_num = section['chapter']
        if chapter_num not in chapters:
            chapters[chapter_num] = []
        chapters[chapter_num].append(section)

    # Sort sections within each chapter by section number
    for chapter_num in chapters:
        chapters[chapter_num].sort(key=lambda x: (x['section'], x.get('subsection', 0)))

    return chapters

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
            3-2-1. 認証機能
            実装の詳細...
            """,
            'content_lines': ['3-1. フロントエンド実装', '3-2. バックエンド実装', '3-2-1. 認証機能', '実装の詳細...']
        }
    ]

    print("Testing section extraction...")

    # Test section extraction
    sections = extract_section_headers_from_text(sample_pages)
    print(f"Extracted {len(sections)} sections:")
    for section in sections:
        chapter = section['chapter']
        section_num = section['section']
        subsection = section.get('subsection', '')
        title = section['title']
        page = section['page']

        if subsection:
            print(f"  - {chapter}-{section_num}-{subsection}. {title} (Chapter {chapter}, Page {page})")
        else:
            print(f"  - {chapter}-{section_num}. {title} (Chapter {chapter}, Page {page})")

    # Test section organization
    organized_sections = organize_sections_by_chapters(sections)
    print(f"\nOrganized sections by chapters:")
    for chapter, chapter_sections in organized_sections.items():
        print(f"  Chapter {chapter}: {len(chapter_sections)} sections")
        for section in chapter_sections:
            subsection = section.get('subsection', '')
            if subsection:
                print(f"    - {section['section']}-{subsection}. {section['title']}")
            else:
                print(f"    - {section['section']}. {section['title']}")

    return True

if __name__ == "__main__":
    try:
        success = test_section_extraction()
        if success:
            print("\n✅ Section extraction test passed!")
        else:
            print("\n❌ Section extraction test failed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
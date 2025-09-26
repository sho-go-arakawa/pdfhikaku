#!/usr/bin/env python
"""
Full integration test for PDF comparison functionality
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from pdfhikaku import (
    PDFExtractor, OutlineDetector, ChapterBasedChunker, 
    TextDiffer, SimilarityCalculator
)

def test_full_pipeline():
    print("=== Full Integration Test ===")
    
    # Create test components
    extractor = PDFExtractor()
    outline_detector = OutlineDetector()
    chapter_chunker = ChapterBasedChunker()
    differ = TextDiffer()
    calculator = SimilarityCalculator()
    
    print("✅ All components created successfully")
    
    # Simulate PDF pages with proper heading extraction
    pages_a = [
        {
            'page_num': 1, 
            'text': '第1章 はじめに\nこれはサンプルテキストです。',
            'headings': [
                {'text': '第1章 はじめに', 'level': 1, 'page': 1}
            ]
        },
        {
            'page_num': 2, 
            'text': '第2章 詳細\n詳細な内容を記述します。',
            'headings': [
                {'text': '第2章 詳細', 'level': 1, 'page': 2}
            ]
        }
    ]
    
    pages_b = [
        {
            'page_num': 1, 
            'text': '第1章 はじめに\nこれは変更されたテキストです。',
            'headings': [
                {'text': '第1章 はじめに', 'level': 1, 'page': 1}
            ]
        },
        {
            'page_num': 2, 
            'text': '第2章 詳細\n異なる内容を記述します。',
            'headings': [
                {'text': '第2章 詳細', 'level': 1, 'page': 2}
            ]
        }
    ]
    
    print("✅ Test data created")
    
    # Create outline from headings
    outline_a = outline_detector.create_outline_from_headings(pages_a)
    outline_b = outline_detector.create_outline_from_headings(pages_b)
    
    print(f"✅ Outlines created: A={len(outline_a)}, B={len(outline_b)}")
    
    # Get chapter structures using the new method
    chapters_a = outline_detector.get_chapter_structure(outline_a)
    chapters_b = outline_detector.get_chapter_structure(outline_b)
    
    print(f"✅ Chapter structures: A={chapters_a}")
    print(f"                       B={chapters_b}")
    
    # Create chapter chunks
    chapter_chunks_a = chapter_chunker.create_chapter_chunks(pages_a, chapters_a)
    chapter_chunks_b = chapter_chunker.create_chapter_chunks(pages_b, chapters_b)
    
    print(f"✅ Chapter chunks: A={len(chapter_chunks_a)}, B={len(chapter_chunks_b)}")
    
    # Test text comparison for each chapter
    for i, (chunk_a, chunk_b) in enumerate(zip(chapter_chunks_a, chapter_chunks_b)):
        print(f"\n--- Testing Chapter {i+1} ---")
        
        # Extract text
        text_a = "\n".join([p.get('text', '') for p in chunk_a.get('pages', [])])
        text_b = "\n".join([p.get('text', '') for p in chunk_b.get('pages', [])])
        
        print(f"Text A ({len(text_a)} chars): {text_a[:50]}...")
        print(f"Text B ({len(text_b)} chars): {text_b[:50]}...")
        
        # Generate diff
        html_a, html_b, stats = differ.generate_diff(text_a, text_b)
        
        print(f"Diff stats: {stats}")
        
        # Test detailed diff
        detailed_diff = differ.generate_detailed_diff(text_a, text_b)
        overall_stats = detailed_diff.get('overall_stats', {})
        
        print(f"Overall similarity: {overall_stats.get('avg_similarity', 0) * 100:.1f}%")
        
        # Calculate similarity
        overall_similarity = calculator.calculate_overall_similarity([stats])
        
        print(f"Calculator similarity: {overall_similarity:.2f}%")
    
    print("\n🎉 Full integration test completed successfully!")
    print("✅ The get_chapter_structure error has been fixed!")
    
if __name__ == "__main__":
    test_full_pipeline()

#!/usr/bin/env python
"""
Direct test of chapter-based comparison without Streamlit
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from pdfhikaku import ChapterBasedChunker, TextDiffer, OutlineDetector

def test_chapter_comparison():
    print("=== Chapter-based Text Comparison Test ===")
    
    # Create sample chapter data structure
    pages_a = [
        {
            'page_num': 1,
            'text': '第1章 はじめに\nこれはPDF Aのテキストです。\n内容について説明します。'
        },
        {
            'page_num': 2,  
            'text': '第2章 詳細\n詳細な内容を記述します。\nより複雑な説明をここに書きます。'
        }
    ]
    
    pages_b = [
        {
            'page_num': 1,
            'text': '第1章 はじめに\nこれはPDF Bのテキストです。\n内容について説明します。'
        },
        {
            'page_num': 2,
            'text': '第2章 詳細\n詳細な内容を記述します。\n異なる説明をここに記載します。'
        }
    ]
    
    # Create chapter structures
    chapters_a = {
        '1': {
            'title': 'はじめに',
            'pages': [1]
        },
        '2': {
            'title': '詳細', 
            'pages': [2]
        }
    }
    
    chapters_b = {
        '1': {
            'title': 'はじめに',
            'pages': [1]
        },
        '2': {
            'title': '詳細',
            'pages': [2]  
        }
    }
    
    # Test chapter chunker
    chunker = ChapterBasedChunker()
    print("Creating chapter chunks...")
    
    chunks_a = chunker.create_chapter_chunks(pages_a, chapters_a)
    chunks_b = chunker.create_chapter_chunks(pages_b, chapters_b)
    
    print(f"Chunks A: {len(chunks_a)}")
    print(f"Chunks B: {len(chunks_b)}")
    
    # Test text differ
    differ = TextDiffer()
    
    for i, (chunk_a, chunk_b) in enumerate(zip(chunks_a, chunks_b)):
        print(f"\n--- Chapter {i+1} Comparison ---")
        print(f"Chapter A: {chunk_a.get('chapter_title', 'Unknown')}")
        print(f"Chapter B: {chunk_b.get('chapter_title', 'Unknown')}")
        
        # Extract text from chunks
        text_a = "\n".join([page.get('text', '') for page in chunk_a.get('pages', [])])
        text_b = "\n".join([page.get('text', '') for page in chunk_b.get('pages', [])])
        
        print(f"Text A length: {len(text_a)}")
        print(f"Text B length: {len(text_b)}")
        
        if text_a and text_b:
            # Generate diff
            html_a, html_b, stats = differ.generate_diff(text_a, text_b)
            print(f"Diff stats: {stats}")
            
            # Generate detailed diff
            detailed = differ.generate_detailed_diff(text_a, text_b)
            print(f"Detailed stats: {detailed.get('overall_stats', {})}")
            
            print(f"HTML A sample: {html_a[:100]}...")
            print(f"HTML B sample: {html_b[:100]}...")
        else:
            print("No text content found")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_chapter_comparison()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple test script to verify RAG search functionality
"""

import os
from utils.pinecone_io import PineconeManager
from utils.text_chunk import TextChunker
import openai


def test_search():
    """Test the search functionality"""
    
    # Initialize OpenAI client for embedding
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Initialize Pinecone manager
    pinecone_manager = PineconeManager()
    
    # Test queries
    test_queries = [
        "フレックスタイムのコアタイムは何時ですか？",
        "有給休暇の申請方法",
        "休憩時間について",
        "時間外勤務の上限"
    ]
    
    print("🔍 RAG検索テスト\n")
    
    for query in test_queries:
        print(f"質問: {query}")
        
        # Create embedding for query
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_vector = response.data[0].embedding
        
        # Search similar chunks
        results = pinecone_manager.search_similar(query_vector, top_k=3)
        
        print(f"検索結果 ({len(results)}件):")
        for i, result in enumerate(results):
            print(f"  {i+1}. スコア: {result['score']:.3f}")
            print(f"     テキスト: {result['text'][:100]}...")
            print()
        
        print("-" * 50)


if __name__ == "__main__":
    test_search()
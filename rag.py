#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
from typing import List, Dict, Any
import openai
from tenacity import retry, wait_exponential, stop_after_attempt

from utils.text_chunk import TextChunker
from utils.pinecone_io import PineconeManager


class RAGProcessor:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.embedding_model = "text-embedding-3-small"
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def create_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """テキストリストをOpenAI APIで埋め込みベクトルに変換"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Creating embeddings for batch {i//batch_size + 1}: {len(batch)} texts")
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def process_file(self, file_path: str, chunker: TextChunker) -> List[Dict[str, Any]]:
        """ファイルを読み込み、チャンク化して埋め込みベクトルを作成"""
        print(f"Reading file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print("Creating chunks...")
        chunks = chunker.create_chunks(text)
        print(f"Created {len(chunks)} chunks")
        
        # テキストリストを抽出
        texts = [chunk["text"] for chunk in chunks]
        
        print("Creating embeddings...")
        embeddings = self.create_embeddings(texts)
        
        # チャンクにベクトルを追加
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "source": chunk["metadata"]["source"],
                    "tokens": chunk["metadata"]["tokens"],
                    "char_start": chunk["metadata"]["char_start"],
                    "char_end": chunk["metadata"]["char_end"]
                }
            })
        
        return vectors


def main():
    parser = argparse.ArgumentParser(description="RAG processing for company policies")
    parser.add_argument("--input", required=True, help="Input text file path")
    parser.add_argument("--max-tokens", type=int, default=300, help="Maximum tokens per chunk")
    parser.add_argument("--overlap-tokens", type=int, default=80, help="Overlap tokens between chunks")
    parser.add_argument("--min-chars", type=int, default=40, help="Minimum characters for sentence merging")
    parser.add_argument("--push-pinecone", action="store_true", help="Push vectors to Pinecone")
    parser.add_argument("--clear-namespace", action="store_true", help="Clear Pinecone namespace before uploading")
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    # チャンカーの初期化
    chunker = TextChunker(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chars=args.min_chars
    )
    
    # RAGプロセッサーの初期化
    processor = RAGProcessor()
    
    try:
        # ファイル処理
        vectors = processor.process_file(args.input, chunker)
        
        print(f"Generated {len(vectors)} vectors")
        
        # ベクトル次元数を動的に取得
        vector_dimension = len(vectors[0]["values"]) if vectors else 0
        print(f"Vector dimension: {vector_dimension}")
        
        # Pineconeへのアップロード
        if args.push_pinecone:
            print("Connecting to Pinecone...")
            
            # Create a temporary Pinecone client to check/create index
            from pinecone import Pinecone
            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            index_name = os.environ["PINECONE_INDEX_NAME"]
            
            # Check if index exists, if not create it
            try:
                pc.Index(index_name)
                print(f"Index '{index_name}' already exists.")
            except:
                print(f"Creating index '{index_name}' with dimension {vector_dimension}...")
                from pinecone import ServerlessSpec
                
                pc.create_index(
                    name=index_name,
                    dimension=vector_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # Wait for index to be ready
                import time
                print("Waiting for index to be ready", end="")
                while True:
                    try:
                        pc.Index(index_name)
                        print(f"\nIndex '{index_name}' is ready!")
                        break
                    except:
                        print(".", end="", flush=True)
                        time.sleep(2)
            
            # Now create the manager
            pinecone_manager = PineconeManager()
            
            if args.clear_namespace:
                print("Clearing Pinecone namespace...")
                pinecone_manager.delete_namespace()
            
            print("Uploading vectors to Pinecone...")
            pinecone_manager.upsert_vectors(vectors)
            
            # 統計情報を表示
            stats = pinecone_manager.get_index_stats()
            print(f"Index stats: {stats}")
            
            print("Successfully uploaded to Pinecone!")
        else:
            print("Skipping Pinecone upload (use --push-pinecone to upload)")
        
        print("\nSample chunks:")
        for i, vector in enumerate(vectors[:3]):
            print(f"Chunk {i+1}:")
            print(f"  ID: {vector['id']}")
            print(f"  Tokens: {vector['metadata']['tokens']}")
            print(f"  Text preview: {vector['metadata']['text'][:100]}...")
            print()
    
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
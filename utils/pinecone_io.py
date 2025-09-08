import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from tenacity import retry, wait_exponential, stop_after_attempt


class PineconeManager:
    def __init__(self):
        self.api_key = os.environ["PINECONE_API_KEY"]
        self.index_name = os.environ["PINECONE_INDEX_NAME"]
        self.namespace = os.environ["PINECONE_NAMESPACE"]
        
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check if index exists, if not provide helpful error message
        try:
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            print(f"Error connecting to Pinecone index '{self.index_name}':")
            print(f"  {e}")
            print(f"\nPlease ensure:")
            print(f"1. The index '{self.index_name}' exists in your Pinecone project")
            print(f"2. Your PINECONE_API_KEY is correct")
            print(f"3. Your PINECONE_INDEX_NAME environment variable is set correctly")
            
            # List available indexes
            try:
                indexes = self.pc.list_indexes()
                if indexes:
                    print(f"\nAvailable indexes in your project:")
                    for idx in indexes:
                        print(f"  - {idx['name']}")
                else:
                    print(f"\nNo indexes found in your project. Please create one first.")
            except Exception as list_error:
                print(f"Could not list indexes: {list_error}")
            
            raise
    
    def create_index_if_not_exists(self, dimension: int = 1536, metric: str = "cosine"):
        """Create index if it doesn't exist"""
        try:
            # Check if index already exists
            self.index = self.pc.Index(self.index_name)
            print(f"Index '{self.index_name}' already exists.")
            return
        except:
            # Index doesn't exist, create it
            pass
        
        print(f"Creating index '{self.index_name}' with dimension {dimension}...")
        
        from pinecone import ServerlessSpec
        
        self.pc.create_index(
            name=self.index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        
        # Wait for index to be ready
        import time
        print("Waiting for index to be ready...")
        while True:
            try:
                self.index = self.pc.Index(self.index_name)
                stats = self.index.describe_index_stats()
                print(f"Index '{self.index_name}' is ready!")
                break
            except:
                print(".", end="", flush=True)
                time.sleep(2)
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def upsert_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """ベクトルをPineconeにバッチでアップサート"""
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            # Pinecone形式に変換
            upsert_data = []
            for vector in batch:
                upsert_data.append({
                    "id": vector["id"],
                    "values": vector["values"],
                    "metadata": vector["metadata"]
                })
            
            self.index.upsert(
                vectors=upsert_data,
                namespace=self.namespace
            )
            
            print(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """類似ベクトルを検索"""
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True
        )
        
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "text": match["metadata"]["text"],
                "metadata": match["metadata"]
            }
            for match in results["matches"]
        ]
    
    def get_index_stats(self) -> Dict[str, Any]:
        """インデックスの統計情報を取得"""
        return self.index.describe_index_stats()
    
    def delete_namespace(self) -> None:
        """名前空間内の全データを削除"""
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            print(f"Deleted all vectors in namespace: {self.namespace}")
        except Exception as e:
            if "Namespace not found" in str(e):
                print(f"Namespace '{self.namespace}' is empty or doesn't exist yet.")
            else:
                raise
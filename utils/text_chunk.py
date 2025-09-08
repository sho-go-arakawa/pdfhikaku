import tiktoken
import re
from typing import List, Dict, Any


class TextChunker:
    def __init__(self, max_tokens: int = 300, overlap_tokens: int = 80, min_chars: int = 40):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chars = min_chars
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """テキストを文単位で分割"""
        sentences = re.split(r'(?<=[。！？])\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def count_tokens(self, text: str) -> int:
        """テキストのトークン数をカウント"""
        return len(self.encoding.encode(text))
    
    def merge_short_sentences(self, sentences: List[str]) -> List[str]:
        """短すぎる文を前後とマージ"""
        merged = []
        current = ""
        
        for sentence in sentences:
            if current and len(current + sentence) < self.min_chars:
                current += sentence
            else:
                if current:
                    merged.append(current)
                current = sentence
        
        if current:
            merged.append(current)
        
        return merged
    
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """テキストをチャンク化"""
        sentences = self.split_into_sentences(text)
        sentences = self.merge_short_sentences(sentences)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        char_start = 0
        
        for i, sentence in enumerate(sentences):
            test_chunk = current_chunk + sentence
            
            if self.count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunk_id = f"policy-{chunk_index:05d}"
                    char_end = char_start + len(current_chunk)
                    
                    chunks.append({
                        "id": chunk_id,
                        "text": current_chunk.strip(),
                        "metadata": {
                            "chunk_index": chunk_index,
                            "source": "company_policies",
                            "tokens": self.count_tokens(current_chunk),
                            "char_start": char_start,
                            "char_end": char_end
                        }
                    })
                    
                    # オーバーラップを考慮して次のチャンクの開始位置を調整
                    overlap_text = self._get_overlap_text(current_chunk)
                    char_start = char_end - len(overlap_text) if overlap_text else char_end
                    current_chunk = overlap_text + sentence
                    chunk_index += 1
                else:
                    current_chunk = sentence
        
        # 最後のチャンクを追加
        if current_chunk.strip():
            chunk_id = f"policy-{chunk_index:05d}"
            char_end = char_start + len(current_chunk)
            
            chunks.append({
                "id": chunk_id,
                "text": current_chunk.strip(),
                "metadata": {
                    "chunk_index": chunk_index,
                    "source": "company_policies",
                    "tokens": self.count_tokens(current_chunk),
                    "char_start": char_start,
                    "char_end": char_end
                }
            })
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """オーバーラップ用のテキストを取得"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.overlap_tokens:
            return text
        
        overlap_tokens = tokens[-self.overlap_tokens:]
        overlap_text = self.encoding.decode(overlap_tokens)
        
        # 文の境界で切り取る
        sentences = self.split_into_sentences(overlap_text)
        if len(sentences) > 1:
            return "".join(sentences[1:])  # 最初の不完全な文を除く
        
        return overlap_text
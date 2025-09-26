"""Utility functions for progress tracking, memory management, and caching."""

import os
import gc
import logging
import json
import pickle
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import hashlib
import time
from contextlib import contextmanager

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Track and display progress for long-running operations."""

    def __init__(self):
        self.status_text = None
        self.progress_bar = None
        self.start_time = None

    def start(self, status_text=None, progress_bar=None):
        """Initialize progress tracking."""
        self.status_text = status_text
        self.progress_bar = progress_bar
        self.start_time = time.time()

    def update(self, progress: float, message: str):
        """Update progress display."""
        if self.progress_bar:
            self.progress_bar.progress(progress / 100)

        if self.status_text:
            elapsed = time.time() - self.start_time if self.start_time else 0
            self.status_text.text(f"{message} (経過時間: {elapsed:.1f}秒)")

        logger.info(f"Progress: {progress:.1f}% - {message}")

    def complete(self, message: str = "完了"):
        """Mark progress as complete."""
        if self.progress_bar:
            self.progress_bar.progress(1.0)

        if self.status_text:
            elapsed = time.time() - self.start_time if self.start_time else 0
            self.status_text.text(f"{message} (総時間: {elapsed:.1f}秒)")

        logger.info(f"Completed: {message}")

class MemoryManager:
    """Manage memory usage and optimization for large file processing."""

    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        if HAS_PSUTIL:
            self.process = psutil.Process()
        else:
            self.process = None

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if not HAS_PSUTIL or not self.process:
            return {
                'rss_mb': 0.0,
                'vms_mb': 0.0,
                'percent': 0.0,
                'available_mb': 1024.0  # Assume 1GB available
            }

        memory_info = self.process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }

    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        usage = self.get_memory_usage()
        return usage['rss_mb'] < self.max_memory_mb

    @contextmanager
    def memory_monitor(self, operation_name: str):
        """Context manager to monitor memory usage during operation."""
        start_memory = self.get_memory_usage()
        start_time = time.time()

        logger.info(f"Starting {operation_name} - Memory: {start_memory['rss_mb']:.1f}MB")

        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            elapsed = time.time() - start_time
            memory_delta = end_memory['rss_mb'] - start_memory['rss_mb']

            logger.info(
                f"Completed {operation_name} - "
                f"Memory: {end_memory['rss_mb']:.1f}MB (+{memory_delta:+.1f}MB) "
                f"Time: {elapsed:.1f}s"
            )

    def optimize_memory(self):
        """Force garbage collection and memory optimization."""
        before = self.get_memory_usage()
        gc.collect()
        after = self.get_memory_usage()

        freed_mb = before['rss_mb'] - after['rss_mb']
        if freed_mb > 1:  # Only log if significant memory was freed
            logger.info(f"Memory optimization freed {freed_mb:.1f}MB")

    def is_memory_critical(self) -> bool:
        """Check if memory usage is critically high."""
        usage = self.get_memory_usage()
        return (usage['rss_mb'] > self.max_memory_mb * 0.9 or
                usage['available_mb'] < 500)  # Less than 500MB available

class FileCache:
    """Caching system for processed data to avoid recomputation."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, cache_key: str, file_type: str = "json") -> Optional[Any]:
        """Get data from cache."""
        cache_file = self.cache_dir / f"{cache_key}.{file_type}"

        if not cache_file.exists():
            return None

        try:
            if file_type == "json":
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_type == "pickle":
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()

        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None

    def set(self, cache_key: str, data: Any, file_type: str = "json"):
        """Store data in cache."""
        cache_file = self.cache_dir / f"{cache_key}.{file_type}"

        try:
            if file_type == "json":
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            elif file_type == "pickle":
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(str(data))

        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def has(self, cache_key: str, file_type: str = "json") -> bool:
        """Check if cache entry exists."""
        cache_file = self.cache_dir / f"{cache_key}.{file_type}"
        return cache_file.exists()

    def clear(self, pattern: str = "*"):
        """Clear cache entries matching pattern."""
        import glob
        cache_files = glob.glob(str(self.cache_dir / f"{pattern}*"))
        cleared_count = 0

        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                cleared_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        logger.info(f"Cleared {cleared_count} cache entries")

    def get_cache_size(self) -> Dict[str, Any]:
        """Get cache directory size and file count."""
        total_size = 0
        file_count = 0

        for cache_file in self.cache_dir.rglob('*'):
            if cache_file.is_file():
                file_count += 1
                total_size += cache_file.stat().st_size

        return {
            'file_count': file_count,
            'total_size_mb': total_size / 1024 / 1024,
            'cache_dir': str(self.cache_dir)
        }

def chunked_processing(items: list, chunk_size: int,
                      process_func: Callable,
                      progress_callback: Optional[Callable] = None) -> list:
    """Process items in chunks to manage memory usage."""
    results = []
    total_items = len(items)

    for i in range(0, total_items, chunk_size):
        chunk = items[i:i + chunk_size]
        chunk_results = process_func(chunk)
        results.extend(chunk_results)

        if progress_callback:
            progress = min(100, (i + len(chunk)) / total_items * 100)
            progress_callback(progress, f"Processed {i + len(chunk)}/{total_items} items")

        # Memory optimization after each chunk
        gc.collect()

    return results

def safe_file_operation(operation: Callable, *args, max_retries: int = 3, **kwargs):
    """Perform file operation with retry logic."""
    for attempt in range(max_retries):
        try:
            return operation(*args, **kwargs)
        except (IOError, OSError) as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"File operation failed (attempt {attempt + 1}): {e}")
            time.sleep(0.5 * (attempt + 1))  # Exponential backoff

def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """Calculate hash of a file."""
    hash_algo = hashlib.new(algorithm)

    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hash_algo.update(chunk)

    return hash_algo.hexdigest()

def estimate_processing_time(file_size_mb: float, base_time_per_mb: float = 2.0) -> float:
    """Estimate processing time based on file size."""
    # Simple linear estimation - could be made more sophisticated
    return file_size_mb * base_time_per_mb

def format_bytes(size_bytes: int) -> str:
    """Format byte size in human readable format."""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s} {size_names[i]}"

def create_temp_directory(prefix: str = "pdfhikaku_") -> str:
    """Create a temporary directory."""
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    logger.info(f"Created temporary directory: {temp_dir}")
    return temp_dir

def cleanup_temp_directory(temp_dir: str):
    """Clean up temporary directory."""
    import shutil
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

class BatchProcessor:
    """Process items in batches with memory management."""

    def __init__(self, batch_size: int = 32, max_memory_mb: int = 1024):
        self.batch_size = batch_size
        self.memory_manager = MemoryManager(max_memory_mb)
        self.cache = FileCache()

    def process_batches(self, items: list, process_func: Callable,
                       cache_prefix: Optional[str] = None,
                       progress_callback: Optional[Callable] = None) -> list:
        """Process items in batches with caching and memory management."""
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(items), self.batch_size):
            batch = items[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1

            # Check cache if enabled
            if cache_prefix:
                cache_key = self.cache.get_cache_key(cache_prefix, batch_idx, len(batch))
                cached_result = self.cache.get(cache_key, "pickle")
                if cached_result:
                    results.extend(cached_result)
                    if progress_callback:
                        progress = (batch_num / total_batches) * 100
                        progress_callback(progress, f"Loaded batch {batch_num}/{total_batches} from cache")
                    continue

            # Process batch
            with self.memory_manager.memory_monitor(f"Batch {batch_num}"):
                batch_results = process_func(batch)
                results.extend(batch_results)

                # Cache results if enabled
                if cache_prefix:
                    cache_key = self.cache.get_cache_key(cache_prefix, batch_idx, len(batch))
                    self.cache.set(cache_key, batch_results, "pickle")

            # Memory optimization
            if self.memory_manager.is_memory_critical():
                self.memory_manager.optimize_memory()

            # Progress update
            if progress_callback:
                progress = (batch_num / total_batches) * 100
                progress_callback(progress, f"Processed batch {batch_num}/{total_batches}")

        return results

# Global instances
memory_manager = MemoryManager()
file_cache = FileCache()
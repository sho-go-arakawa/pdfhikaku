"""Configuration management for PDFHikaku application."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

class Config:
    """Configuration manager that loads settings from YAML and environment variables."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config_data = self._load_config()
        self._setup_logging()

    def _find_config_file(self) -> str:
        """Find configuration file in standard locations."""
        possible_paths = [
            "configs/config.yaml",
            "config.yaml",
            os.path.expanduser("~/.pdfhikaku/config.yaml")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Use example config if no config found
        example_path = "configs/config.yaml.example"
        if os.path.exists(example_path):
            return example_path

        raise FileNotFoundError("No configuration file found")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not HAS_YAML:
            logging.warning("PyYAML not available, using default configuration")
            return self._get_default_config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()

    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        log_file = log_config.get('file', 'logs/app.log')
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self.config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_extract_config(self) -> Dict[str, Any]:
        """Get extraction configuration."""
        return {
            'engine_priority': self.get('extract.engine_priority', ['pymupdf', 'pdfminer', 'ocr']),
            'ocr_lang': self.get('extract.ocr_lang', 'jpn+eng'),
            'max_pages': self.get('extract.max_pages'),
            'remove_patterns': self.get('extract.remove_patterns', []),
            'tesseract_cmd': os.getenv('TESSERACT_CMD', '/usr/bin/tesseract')
        }

    def get_chunk_config(self) -> Dict[str, Any]:
        """Get chunking configuration."""
        return {
            'max_tokens': self.get('chunk.max_tokens', 800),
            'sentence_split': self.get('chunk.sentence_split', True),
            'overlap_tokens': self.get('chunk.overlap_tokens', 50),
            'min_chunk_size': self.get('chunk.min_chunk_size', 100)
        }

    def get_align_config(self) -> Dict[str, Any]:
        """Get alignment configuration."""
        return {
            'title_weight': self.get('align.title_weight', 0.4),
            'embed_weight': self.get('align.embed_weight', 0.6),
            'string_weight': self.get('align.string_weight', 0.4),
            'exact_threshold': self.get('align.exact_threshold', 0.92),
            'partial_threshold': self.get('align.partial_threshold', 0.75),
            'gap_penalty': self.get('align.gap_penalty', 0.15),
            'embedding_model': self.get('align.embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            'use_openai': self.get('align.use_openai', False),
            'openai_model': self.get('align.openai_model', 'text-embedding-3-small'),
            'openai_api_key': os.getenv('OPENAI_API_KEY')
        }

    def get_perf_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return {
            'workers': int(os.getenv('MAX_WORKERS', self.get('perf.workers', 4))),
            'mmap': self.get('perf.mmap', True),
            'cache_dir': os.getenv('CACHE_DIR', self.get('perf.cache_dir', '.cache')),
            'max_memory_mb': int(os.getenv('MAX_MEMORY_MB', self.get('perf.max_memory_mb', 2048))),
            'chunk_batch_size': self.get('perf.chunk_batch_size', 32)
        }

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return {
            'layout': self.get('ui.layout', 'wide'),
            'sidebar_width': self.get('ui.sidebar_width', 300),
            'max_file_size_mb': self.get('ui.max_file_size_mb', 100),
            'supported_formats': self.get('ui.supported_formats', ['pdf']),
            'theme': self.get('ui.theme', 'light')
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file loading fails."""
        return {
            'extract': {
                'engine_priority': ['pymupdf', 'pdfminer', 'ocr'],
                'ocr_lang': 'jpn+eng',
                'max_pages': None,
                'remove_patterns': []
            },
            'chunk': {
                'max_tokens': 800,
                'sentence_split': True,
                'overlap_tokens': 50,
                'min_chunk_size': 100
            },
            'align': {
                'title_weight': 0.4,
                'embed_weight': 0.6,
                'string_weight': 0.4,
                'exact_threshold': 0.92,
                'partial_threshold': 0.75,
                'gap_penalty': 0.15,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'use_openai': False,
                'openai_model': 'text-embedding-3-small'
            },
            'perf': {
                'workers': 4,
                'mmap': True,
                'cache_dir': '.cache',
                'max_memory_mb': 2048,
                'chunk_batch_size': 32
            },
            'ui': {
                'layout': 'wide',
                'sidebar_width': 300,
                'max_file_size_mb': 100,
                'supported_formats': ['pdf'],
                'theme': 'light'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/app.log'
            }
        }

# Global config instance
config = Config()
"""
BGE-M3 Embedder Module with GPU Acceleration

RTX 4060Ti (8GB VRAM) optimized embedding generation using BAAI/bge-m3 model.
"""

import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGE_M3Embedder:
    """GPU-accelerated BGE-M3 embedding generator."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
        use_fp16: bool = True,
        max_seq_length: int = 512,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize BGE-M3 embedder with GPU optimization.

        Args:
            model_name: Hugging Face model name
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for embedding generation
            use_fp16: Use mixed precision for faster computation
            max_seq_length: Maximum sequence length
            cache_dir: Cache directory for model files
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.max_seq_length = max_seq_length

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Log GPU information
        if self.device == "cuda":
            gpu_info = self._get_gpu_info()
            logger.info(
                f"Using GPU: {gpu_info['name']} with {gpu_info['memory_gb']:.1f}GB VRAM"
            )

            # Adjust batch size based on VRAM
            if gpu_info["memory_gb"] < 6:
                self.batch_size = min(self.batch_size, 16)
                logger.info(
                    f"Adjusted batch size to {self.batch_size} for limited VRAM"
                )
        else:
            logger.warning("GPU not available, using CPU (will be slower)")

        # Initialize model
        self._initialize_model(cache_dir)

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        if not torch.cuda.is_available():
            return {"name": "N/A", "memory_gb": 0}

        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "memory_gb": props.total_memory / (1024**3),
            "compute_capability": f"{props.major}.{props.minor}",
        }

    def _initialize_model(self, cache_dir: Optional[Path] = None):
        """Initialize the sentence transformer model."""
        try:
            logger.info(f"Loading {self.model_name} model...")

            # Model configuration
            model_kwargs = {
                "torch_dtype": (
                    torch.float16
                    if self.use_fp16 and self.device == "cuda"
                    else torch.float32
                )
            }

            # Load model
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(cache_dir) if cache_dir else None,
            )

            # Set max sequence length
            self.model.max_seq_length = self.max_seq_length

            # Enable eval mode
            self.model.eval()

            # Move to GPU and optimize if available
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                if self.use_fp16:
                    self.model = self.model.half()

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for texts using GPU acceleration.

        Args:
            texts: Single text or list of texts
            batch_size: Override default batch size
            normalize_embeddings: Normalize embeddings to unit length
            show_progress_bar: Show encoding progress
            convert_to_numpy: Convert to numpy array

        Returns:
            Embeddings as numpy array
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.batch_size

        try:
            # Clear GPU cache before encoding
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=convert_to_numpy,
                    device=self.device,
                )

            # Log memory usage
            if self.device == "cuda":
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                logger.debug(f"GPU memory used: {memory_used:.2f}GB")

            return embeddings

        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM, falling back to smaller batch size")
            # Retry with smaller batch
            if batch_size > 1:
                return self.encode(
                    texts,
                    batch_size=batch_size // 2,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=convert_to_numpy,
                )
            else:
                raise

        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

    def encode_batch(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Encode texts in optimized batches.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing

        Returns:
            List of embeddings
        """
        batch_size = batch_size or self.batch_size
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.encode(batch)
            embeddings.extend(batch_embeddings)

            # Periodic cleanup
            if self.device == "cuda" and i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                gc.collect()

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()

    def similarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between embeddings.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Similarity scores
        """
        # Convert to torch tensors
        emb1 = torch.from_numpy(embeddings1).to(self.device)
        emb2 = torch.from_numpy(embeddings2).to(self.device)

        # Normalize
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        # Calculate cosine similarity
        similarity = torch.mm(emb1, emb2.t())

        return similarity.cpu().numpy()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if self.device != "cuda":
            return {"allocated_gb": 0, "cached_gb": 0}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "cached_gb": torch.cuda.memory_reserved() / (1024**3),
        }

    def clear_cache(self):
        """Clear GPU cache."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def __del__(self):
        """Cleanup on deletion."""
        self.clear_cache()

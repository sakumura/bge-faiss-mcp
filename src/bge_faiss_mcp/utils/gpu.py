"""
GPU Optimizer Module

Utilities for GPU memory management and performance optimization.
"""

import torch
import psutil
import logging
from typing import Dict, Any, Optional
import gc
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUOptimizer:
    """GPU optimization and monitoring utilities."""

    def __init__(self, gpu_id: int = 0):
        """
        Initialize GPU optimizer.

        Args:
            gpu_id: GPU device ID to monitor
        """
        self.gpu_id = gpu_id
        self.cuda_available = torch.cuda.is_available()

        if self.cuda_available:
            self.device = torch.device(f"cuda:{gpu_id}")
            self.gpu_properties = torch.cuda.get_device_properties(gpu_id)
            logger.info(f"GPU Optimizer initialized for {self.gpu_properties.name}")
        else:
            logger.warning("CUDA not available, GPU optimization disabled")

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information."""
        if not self.cuda_available:
            return {"cuda_available": False}

        return {
            "cuda_available": True,
            "device_name": self.gpu_properties.name,
            "total_memory_gb": self.gpu_properties.total_memory / (1024**3),
            "compute_capability": f"{self.gpu_properties.major}.{self.gpu_properties.minor}",
            "multi_processor_count": self.gpu_properties.multi_processor_count,
            "cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
        }

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if not self.cuda_available:
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated(self.gpu_id) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(self.gpu_id) / (1024**3),
            "free_gb": (
                self.gpu_properties.total_memory
                - torch.cuda.memory_allocated(self.gpu_id)
            )
            / (1024**3),
            "utilization_percent": (
                torch.cuda.memory_allocated(self.gpu_id)
                / self.gpu_properties.total_memory
            )
            * 100,
        }

    def optimize_batch_size(
        self,
        model_memory_gb: float = 2.0,
        sample_memory_mb: float = 10.0,
        safety_factor: float = 0.8,
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            model_memory_gb: Estimated model memory in GB
            sample_memory_mb: Memory per sample in MB
            safety_factor: Safety margin (0-1)

        Returns:
            Recommended batch size
        """
        if not self.cuda_available:
            return 1

        memory_stats = self.get_memory_stats()
        available_gb = memory_stats["free_gb"] * safety_factor

        # Subtract model memory
        available_for_data = max(0, available_gb - model_memory_gb)

        # Calculate batch size
        batch_size = int((available_for_data * 1024) / sample_memory_mb)

        # Apply constraints based on GPU
        if "4060" in self.gpu_properties.name:
            # RTX 4060 Ti specific optimizations
            batch_size = min(batch_size, 64)  # Max batch size
            batch_size = max(batch_size, 1)  # Min batch size

        logger.info(
            f"Recommended batch size: {batch_size} (Available: {available_gb:.2f}GB)"
        )
        return batch_size

    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        if self.cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.debug("GPU cache cleared")

    def set_memory_fraction(self, fraction: float = 0.9):
        """
        Set maximum GPU memory fraction to use.

        Args:
            fraction: Fraction of GPU memory to use (0-1)
        """
        if self.cuda_available:
            torch.cuda.set_per_process_memory_fraction(fraction, self.gpu_id)
            logger.info(f"GPU memory fraction set to {fraction}")

    def enable_mixed_precision(self):
        """Enable mixed precision training/inference."""
        if self.cuda_available:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Mixed precision enabled")

    def get_system_memory(self) -> Dict[str, float]:
        """Get system RAM statistics."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent,
        }

    def monitor_memory(self, tag: str = ""):
        """
        Log current memory usage.

        Args:
            tag: Optional tag for the log message
        """
        if self.cuda_available:
            gpu_stats = self.get_memory_stats()
            logger.info(
                f"[{tag}] GPU Memory: {gpu_stats['allocated_gb']:.2f}GB / {self.gpu_properties.total_memory / (1024**3):.2f}GB"
            )

        sys_stats = self.get_system_memory()
        logger.info(
            f"[{tag}] System RAM: {sys_stats['used_gb']:.2f}GB / {sys_stats['total_gb']:.2f}GB"
        )

    def profile_function(self, func, *args, **kwargs):
        """
        Profile a function's GPU memory usage.

        Args:
            func: Function to profile
            *args, **kwargs: Function arguments

        Returns:
            Function result and memory statistics
        """
        if not self.cuda_available:
            return func(*args, **kwargs), {}

        # Clear cache and get initial memory
        self.clear_cache()
        start_memory = torch.cuda.memory_allocated(self.gpu_id)

        # Run function
        result = func(*args, **kwargs)

        # Get memory after execution
        end_memory = torch.cuda.memory_allocated(self.gpu_id)
        peak_memory = torch.cuda.max_memory_allocated(self.gpu_id)

        stats = {
            "memory_used_mb": (end_memory - start_memory) / (1024**2),
            "peak_memory_mb": peak_memory / (1024**2),
            "current_memory_mb": end_memory / (1024**2),
        }

        logger.info(f"Function profiled: {func.__name__}")
        logger.info(
            f"Memory used: {stats['memory_used_mb']:.2f}MB, Peak: {stats['peak_memory_mb']:.2f}MB"
        )

        return result, stats

    def auto_select_device(self) -> str:
        """
        Automatically select best available device.

        Returns:
            Device string ('cuda:0', 'cpu', etc.)
        """
        if not self.cuda_available:
            logger.info("Using CPU (CUDA not available)")
            return "cpu"

        # Check available memory
        memory_stats = self.get_memory_stats()
        if memory_stats["free_gb"] < 1.0:
            logger.warning(
                f"Low GPU memory ({memory_stats['free_gb']:.2f}GB free), using CPU"
            )
            return "cpu"

        logger.info(f"Using GPU: {self.gpu_properties.name}")
        return f"cuda:{self.gpu_id}"

    def warmup_gpu(self):
        """Warm up GPU with dummy operations."""
        if not self.cuda_available:
            return

        logger.info("Warming up GPU...")
        dummy = torch.randn(1000, 1000, device=self.device)
        for _ in range(10):
            dummy = torch.mm(dummy, dummy)
        del dummy
        self.clear_cache()
        logger.info("GPU warmup complete")

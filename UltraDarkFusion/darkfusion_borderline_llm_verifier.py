#!/usr/bin/env python3
"""
LLM-based borderline case verification using Qwen2-VL or LLaVA.

For DINOv3 similarity in the 10-70% range (ambiguous), uses vision LLM to:
1. Verify if label is correct for the object
2. Provide confidence (high/medium/low)
3. Suggest correct class if wrong

Designed to run locally with Ollama for privacy and speed.
Auto-setup: Automatically detects, installs, and starts Ollama if needed.
"""

import os
import sys
import json
import logging
import base64
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import cv2
import numpy as np

try:
    from darkfusion_ollama_auto_setup import ensure_ollama_ready, check_ollama_running
except ImportError:
    # Fallback if auto-setup module not available
    def ensure_ollama_ready(silent=True):
        return False, None
    def check_ollama_running():
        return False

logger = logging.getLogger(__name__)


class BorderlineLLMVerifier:
    """Verify borderline FP candidates using local vision LLM."""
    
    def __init__(self, 
                 ollama_host: str = "http://localhost:11434",
                 model: str = "qwen2-vl:4b",
                 fallback_model: str = "llava:7b",
                 cache_dir: Optional[str] = None,
                 enabled: bool = True):
        """Initialize LLM verifier.
        
        Args:
            ollama_host: Ollama API endpoint
            model: Primary model to use (qwen2-vl:4b)
            fallback_model: Fallback if primary unavailable (llava:7b)
            cache_dir: Cache directory for results
            enabled: Whether to enable LLM verification
        """
        self.ollama_host = ollama_host
        self.model = model
        self.fallback_model = fallback_model
        self.enabled = enabled
        self.cache_dir = Path(cache_dir or ".darkfusion_llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.available_model = None
        self.health_checked = False
        self._model_cache = {}
        
    def health_check(self) -> bool:
        """Check if Ollama is running and models available.
        
        Auto-attempts setup if not running.
        
        Returns:
            bool: True if at least one model is available
        """
        if self.health_checked:
            return self.available_model is not None
        
        self.health_checked = True
        
        if not self.enabled:
            logger.info("LLM verification disabled")
            return False
        
        # First check: Is Ollama already running?
        try:
            response = requests.get(
                f"{self.ollama_host}/api/tags",
                timeout=2
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check which models are available
                for candidate in [self.model, self.fallback_model]:
                    if any(candidate in name for name in model_names):
                        self.available_model = candidate
                        logger.info(f"LLM verifier using model: {self.available_model}")
                        return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception:
            pass
        
        # Second attempt: Auto-setup Ollama
        logger.info("Attempting automatic Ollama setup...")
        try:
            success, model = ensure_ollama_ready(silent=True)
            if success and model:
                self.available_model = model
                logger.info(f"LLM verifier auto-setup successful, using: {model}")
                return True
        except Exception as e:
            logger.debug(f"Auto-setup failed: {e}")
        
        # Fallback: Log why unavailable
        logger.debug(
            f"LLM verification unavailable. "
            f"Install Ollama: https://ollama.ai, then pull model: ollama pull qwen2-vl:4b"
        )
        return False
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for API.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Base64 encoded image
        """
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Could not encode image {image_path}: {e}")
            return ""
    
    def verify_borderline_candidate(self, 
                                    image_path: str,
                                    class_name: str,
                                    bounds: Optional[tuple] = None) -> Dict[str, Any]:
        """Verify if borderline candidate is correctly labeled.
        
        Args:
            image_path: Path to image file
            class_name: Labeled class name
            bounds: Bounding box (x1, y1, x2, y2) to crop object
        
        Returns:
            dict with:
                - verified: bool, whether label is correct
                - confidence: "high"/"medium"/"low"
                - reasoning: explanation
                - suggested_class: alternative if wrong
                - cached: bool, was result cached
        """
        if not self.available_model:
            if not self.health_check():
                return {
                    "verified": None,
                    "confidence": None,
                    "reasoning": "LLM unavailable - Ollama not running",
                    "suggested_class": None,
                    "cached": False,
                    "error": "ollama_not_available",
                }
        
        # Check cache first
        cache_key = self._get_cache_key(image_path, class_name, bounds)
        cached = self._load_from_cache(cache_key)
        if cached:
            cached["cached"] = True
            return cached
        
        # Prepare image
        if not os.path.isfile(image_path):
            return {
                "verified": None,
                "confidence": None,
                "reasoning": f"Image not found: {image_path}",
                "suggested_class": None,
                "cached": False,
                "error": "image_not_found",
            }
        
        # Crop object if bounds provided (in-memory, no temp files)
        image_data = None
        if bounds and len(bounds) == 4:
            try:
                x1, y1, x2, y2 = [int(b) for b in bounds]
                img = cv2.imread(image_path)
                if img is not None:
                    crop = img[max(0, y1):min(img.shape[0], y2),
                               max(0, x1):min(img.shape[1], x2)]
                    if crop.size > 0:
                        # Encode crop directly to base64 (in-memory, no temp file)
                        success, buffer = cv2.imencode('.jpg', crop)
                        if success:
                            image_data = base64.b64encode(buffer).decode("utf-8")
            except Exception as e:
                logger.warning(f"Could not crop image in-memory: {e}")
        
        # Fallback to full image if crop failed
        if not image_data:
            image_data = self._image_to_base64(image_path)
        
        if not image_data:
            return {
                "verified": None,
                "confidence": None,
                "reasoning": "Could not encode image",
                "suggested_class": None,
                "cached": False,
                "error": "image_encode_failed",
            }
        
        # Ask LLM with cropped (or full) image
        try:
            result = self._query_llm_with_data(image_data, class_name)
            # Cache result
            self._save_to_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return {
                "verified": None,
                "confidence": None,
                "reasoning": f"LLM error: {str(e)[:100]}",
                "suggested_class": None,
                "cached": False,
                "error": "llm_error",
            }
    
    def _query_llm_with_data(self, image_b64: str, class_name: str) -> Dict[str, Any]:
        """Query LLM about image classification with base64 data.
        
        Args:
            image_b64: Base64 encoded image data
            class_name: Labeled class to verify
        
        Returns:
            dict with verification result
        """
        if not image_b64:
            return {
                "verified": None,
                "confidence": None,
                "reasoning": "Could not encode image",
                "suggested_class": None,
                "cached": False,
            }
        
        prompt = f"""Analyze this image of a potentially mislabeled object.

The object is labeled as: {class_name}

Your task:
1. Is this object correctly labeled as "{class_name}"? Answer YES or NO.
2. What is your confidence? Answer: HIGH, MEDIUM, or LOW.
3. If wrong, what should it actually be labeled as?

Be concise. Focus on whether the label matches the object you see.

Answer in this exact format:
CORRECT: YES/NO
CONFIDENCE: HIGH/MEDIUM/LOW
SUGGESTION: [class name if wrong, or "none" if correct]
REASONING: [1-2 sentences explaining]"""
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.available_model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "temperature": 0.3,  # Lower temp = more consistent
                    "top_p": 0.8,
                },
                timeout=30  # 30 sec timeout (was 60) - prevent hanging
            )
            
            if response.status_code != 200:
                logger.warning(f"LLM returned status {response.status_code}")
                return {
                    "verified": None,
                    "confidence": None,
                    "reasoning": f"LLM error: {response.status_code}",
                    "suggested_class": None,
                    "cached": False,
                }
            
            # Parse response
            response_text = response.json().get("response", "").strip()
            return self._parse_llm_response(response_text, class_name)
        
        except requests.exceptions.Timeout:
            logger.warning("LLM request timed out")
            return {
                "verified": None,
                "confidence": None,
                "reasoning": "LLM request timed out (model too slow)",
                "suggested_class": None,
                "cached": False,
            }
        except Exception as e:
            logger.error(f"LLM request error: {e}")
            raise
    
    def _query_llm(self, image_path: str, class_name: str) -> Dict[str, Any]:
        """Query LLM about image classification.
        
        Args:
            image_path: Path to image (or crop)
            class_name: Labeled class to verify
        
        Returns:
            dict with verification result
        """
        image_b64 = self._image_to_base64(image_path)
        return self._query_llm_with_data(image_b64, class_name)
    
    def _parse_llm_response(self, response: str, class_name: str) -> Dict[str, Any]:
        """Parse LLM response into structured format.
        
        Args:
            response: Raw LLM response text
            class_name: Original labeled class
        
        Returns:
            dict with parsed result
        """
        lines = response.split("\n")
        correct = None
        confidence = None
        suggestion = None
        reasoning = ""
        
        for line in lines:
            line_lower = line.lower().strip()
            if "correct:" in line_lower:
                correct = "yes" in line_lower
            elif "confidence:" in line_lower:
                if "high" in line_lower:
                    confidence = "high"
                elif "medium" in line_lower:
                    confidence = "medium"
                elif "low" in line_lower:
                    confidence = "low"
            elif "suggestion:" in line_lower:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    suggestion = parts[1].strip()
                    if suggestion.lower() in ["none", "n/a", "-", ""]:
                        suggestion = None
            elif "reasoning:" in line_lower:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    reasoning = parts[1].strip()
        
        return {
            "verified": correct,
            "confidence": confidence or "medium",
            "reasoning": reasoning or response[:200],
            "suggested_class": suggestion,
            "cached": False,
        }
    
    def _get_cache_key(self, image_path: str, class_name: str, bounds: Optional[tuple]) -> str:
        """Generate cache key for result."""
        import hashlib
        key_parts = [
            os.path.basename(image_path),
            class_name,
            str(bounds or ""),
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached result."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save result to cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, "w") as f:
                json.dump(result, f)
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")


def verify_with_llm(image_path: str, 
                    class_name: str,
                    bounds: Optional[tuple] = None,
                    cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """Quick function to verify a single borderline candidate.
    
    Args:
        image_path: Path to image
        class_name: Labeled class
        bounds: Optional bounding box
        cache_dir: Cache directory
    
    Returns:
        Verification result
    """
    verifier = BorderlineLLMVerifier(cache_dir=cache_dir)
    if not verifier.health_check():
        return {
            "verified": None,
            "confidence": None,
            "reasoning": "LLM unavailable",
            "suggested_class": None,
            "cached": False,
            "error": "ollama_not_available",
        }
    return verifier.verify_borderline_candidate(image_path, class_name, bounds)

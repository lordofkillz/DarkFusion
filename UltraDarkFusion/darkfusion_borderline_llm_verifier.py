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
import hashlib
import math
import re
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import cv2
import numpy as np
from darkfusion_shape_verifier import shape_annotation_crop
from darkfusion_ollama_memory import generate as generate_with_memory_tracking

try:
    from darkfusion_ollama_auto_setup import ensure_ollama_ready, check_ollama_running
except ImportError:
    # Fallback if auto-setup module not available
    def ensure_ollama_ready(silent=True):
        return False, None
    def check_ollama_running():
        return False

logger = logging.getLogger(__name__)
LLM_CACHE_VERSION = "borderline-object-crop-v2"
LLM_PROMPT_VERSION = "saved-class-yes-no-v2"
LLM_GENERATION_OPTIONS = {"temperature": 0.3, "top_p": 0.8}


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
        self._available_model_digest = ""
        
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
                    matched = next((item for item in models if candidate in item.get("name", "")), None)
                    if matched:
                        self.available_model = matched["name"]
                        self._available_model_digest = str(matched.get("digest", "") or "")
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
                                    bounds: Optional[tuple] = None, *,
                                    bounds_space: str = "normalized") -> Dict[str, Any]:
        """Verify if borderline candidate is correctly labeled.
        
        Args:
            image_path: Path to image file
            class_name: Labeled class name
            bounds: Normalized xyxy bounds for the saved annotation.
            bounds_space: "pixels" is supported for explicit legacy API callers.
        
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
        
        # Read once: the content hashed for caching is exactly what gets decoded
        # and sent. A deleted/replaced image cannot inherit a stale answer.
        try:
            image_bytes = Path(image_path).read_bytes()
        except OSError:
            return self._unknown("image_not_found", f"Image unavailable: {image_path}")
        try:
            cache_key = self._get_cache_key(
                image_path, class_name, bounds, bounds_space=bounds_space,
                image_bytes=image_bytes,
            )
        except (TypeError, ValueError):
            return self._unknown("invalid_bounds", "Invalid annotation crop bounds")
        cached = self._load_from_cache(cache_key)
        if cached:
            cached["cached"] = True
            return cached
        try:
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return self._unknown("image_decode_failed", "Could not decode image")
            if bounds is not None:
                normalized = tuple(float(value) for value in bounds)
                if bounds_space == "pixels":
                    height, width = image.shape[:2]
                    normalized = (normalized[0] / width, normalized[1] / height,
                                  normalized[2] / width, normalized[3] / height)
                image = shape_annotation_crop(image, normalized)
                if image is None:
                    return self._unknown("invalid_crop", "Annotation crop is empty or too small")
            success, buffer = cv2.imencode(".jpg", image)
            if not success:
                return self._unknown("image_encode_failed", "Could not encode object crop")
            image_data = base64.b64encode(buffer).decode("utf-8")
        except (ValueError, TypeError, cv2.error) as error:
            return self._unknown("image_encode_failed", f"Object crop unavailable: {error}")
        
        # Ask LLM with cropped (or full) image
        try:
            result = self._query_llm_with_data(image_data, class_name)
            if self._cacheable_result(result):
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
            response = generate_with_memory_tracking(
                self.ollama_host,
                {
                    "model": self.available_model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": dict(LLM_GENERATION_OPTIONS),
                },
                timeout=30  # 30 sec timeout (was 60) - prevent hanging
            )
            
            if response.status_code != 200:
                logger.warning(f"LLM returned status {response.status_code}")
                return self._unknown("llm_http_error", f"LLM error: {response.status_code}")
            
            # Parse response
            response_text = response.json().get("response", "").strip()
            return self._parse_llm_response(response_text, class_name)
        
        except requests.exceptions.Timeout:
            logger.warning("LLM request timed out")
            return self._unknown("llm_timeout", "LLM request timed out; object remains unverified")
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
        correct_values = set()
        confidence_values = set()
        invalid_field = False
        suggestion = None
        reasoning = ""
        
        for line in lines:
            line_lower = line.lower().strip().replace("**", "")
            correct_match = re.fullmatch(r"correct\s*:\s*(yes|no)[.!]?", line_lower)
            confidence_match = re.fullmatch(r"confidence\s*:\s*(high|medium|low)[.!]?", line_lower)
            if correct_match:
                correct_values.add(correct_match.group(1) == "yes")
            elif confidence_match:
                confidence_values.add(confidence_match.group(1))
            elif re.match(r"^(correct|confidence)\s*:", line_lower):
                invalid_field = True
            elif line_lower.startswith("suggestion:"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    suggestion = parts[1].strip()
                    if suggestion.lower() in ["none", "n/a", "-", ""]:
                        suggestion = None
            elif line_lower.startswith("reasoning:"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    reasoning = parts[1].strip()
        
        correct = next(iter(correct_values)) if len(correct_values) == 1 else None
        confidence = next(iter(confidence_values)) if len(confidence_values) == 1 else None
        result = {
            "verified": correct,
            "confidence": confidence,
            "reasoning": reasoning or response[:200],
            "suggested_class": suggestion,
            "cached": False,
        }
        if correct is None or confidence is None or invalid_field:
            result["verified"] = None
            result["error"] = "unparseable_answer"
        return result
    
    @staticmethod
    def _unknown(error, reasoning):
        return {"verified": None, "confidence": None, "reasoning": reasoning,
                "suggested_class": None, "cached": False, "error": error}

    @staticmethod
    def _cacheable_result(result):
        return (isinstance(result, dict) and not result.get("error")
                and type(result.get("verified")) is bool
                and result.get("confidence") in ("high", "medium", "low"))

    def _get_cache_key(self, image_path: str, class_name: str, bounds: Optional[tuple], *,
                       bounds_space="normalized", image_bytes=None) -> str:
        """Versioned identity: exact image content, object, model and prompt."""
        if bounds_space not in ("normalized", "pixels"):
            raise ValueError("Unsupported bounds coordinate space")
        normalized_bounds = None
        if bounds is not None:
            normalized_bounds = tuple(float(value) for value in bounds)
            if len(normalized_bounds) != 4 or not all(math.isfinite(value) for value in normalized_bounds):
                raise ValueError("Invalid crop bounds")
        image_path = os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(image_path))))
        if image_bytes is None:
            image_bytes = Path(image_path).read_bytes()
        identity = {
            "version": LLM_CACHE_VERSION, "prompt": LLM_PROMPT_VERSION,
            "image": image_path, "content": hashlib.sha256(image_bytes).hexdigest(),
            "class": str(class_name), "bounds": normalized_bounds, "bounds_space": bounds_space,
            "model": self.available_model or self.model,
            "model_digest": getattr(self, "_available_model_digest", ""),
            "host": self.ollama_host.rstrip("/"), "options": LLM_GENERATION_OPTIONS,
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached result."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    payload = json.load(f)
                if (isinstance(payload, dict) and payload.get("version") == LLM_CACHE_VERSION
                        and self._cacheable_result(payload.get("result"))):
                    return payload["result"]
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save result to cache."""
        if not self._cacheable_result(result):
            return
        temporary = None
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache_dir,
                                             prefix=".llm-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump({"version": LLM_CACHE_VERSION, "result": result}, handle)
            os.replace(temporary, cache_file)
            temporary = None
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Could not remove incomplete LLM cache temporary file")


def verify_with_llm(image_path: str, 
                    class_name: str,
                    bounds: Optional[tuple] = None,
                    cache_dir: Optional[str] = None, *, bounds_space="normalized") -> Dict[str, Any]:
    """Quick function to verify a single borderline candidate.
    
    Args:
        image_path: Path to image
        class_name: Labeled class
        bounds: Optional normalized xyxy bounding box (or explicit bounds_space="pixels")
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
    return verifier.verify_borderline_candidate(image_path, class_name, bounds, bounds_space=bounds_space)

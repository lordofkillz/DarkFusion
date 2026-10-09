#!/usr/bin/env python3
"""
Automatic Ollama setup and model management.

Handles:
- Ollama installation detection
- Auto-download if missing
- Auto-pull Qwen2-VL model
- Service startup
- Silent fallback if anything fails
"""

import os
import sys
import subprocess
import platform
import logging
import time
import requests
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


class OllamaAutoSetup:
    """Automatic Ollama detection and setup."""
    
    OLLAMA_HOST = "http://localhost:11434"
    PRIMARY_MODEL = "qwen2-vl:4b"
    FALLBACK_MODEL = "llava:7b"
    
    def __init__(self, silent: bool = True):
        """Initialize auto-setup.
        
        Args:
            silent: Don't log warnings/errors to console
        """
        self.silent = silent
        self.system = platform.system()
        self.ollama_installed = False
        self.model_available = None
        self.service_running = False
    
    def ensure_ready(self) -> Tuple[bool, str]:
        """Ensure Ollama is running with a model available.
        
        Returns:
            (success: bool, model_name: str or reason for failure)
        """
        # Check if already running
        if self._check_service_health():
            model = self._detect_available_model()
            if model:
                self.service_running = True
                self.model_available = model
                return True, model
        
        # Check if installed
        if not self._check_ollama_installed():
            # Not installed - try automatic install (silent, non-blocking)
            logger.debug("Ollama not found. Attempting automatic installation...")
            if not self._install_ollama():
                # Installation failed, graceful fallback
                return False, "Ollama not available"
        
        self.ollama_installed = True
        
        # Start service
        if self._start_service():
            self.service_running = True
            # Wait for service to be ready
            if self._wait_for_service(max_wait=30):
                model = self._detect_available_model()
                if model:
                    self.model_available = model
                    return True, model
                else:
                    # Try to pull primary model
                    if self._ensure_model_pulled(self.PRIMARY_MODEL):
                        return True, self.PRIMARY_MODEL
                    elif self._ensure_model_pulled(self.FALLBACK_MODEL):
                        return True, self.FALLBACK_MODEL
        
        # Silent failure - system will work without LLM
        return False, "Ollama not available"
    
    def _check_service_health(self) -> bool:
        """Check if Ollama service is running and responding."""
        try:
            response = requests.get(
                f"{self.OLLAMA_HOST}/api/tags",
                timeout=2
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_ollama_installed(self) -> bool:
        """Check if Ollama is installed."""
        try:
            if self.system == "Windows":
                # Check Windows installation paths
                paths = [
                    Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
                    Path("C:/Program Files/Ollama/ollama.exe"),
                ]
                return any(p.exists() for p in paths)
            elif self.system == "Darwin":  # macOS
                return Path("/Applications/Ollama.app/Contents/MacOS/Ollama").exists()
            else:  # Linux
                result = subprocess.run(
                    ["which", "ollama"],
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0
        except Exception:
            return False
    
    def _install_ollama(self) -> bool:
        """Automatically download and install Ollama (non-blocking)."""
        try:
            if self.system == "Windows":
                return self._install_ollama_windows()
            elif self.system == "Darwin":
                return self._install_ollama_macos()
            else:
                return self._install_ollama_linux()
        except Exception as e:
            if not self.silent:
                logger.debug(f"Ollama auto-install failed: {e}")
            return False
    
    def _install_ollama_windows(self) -> bool:
        """Download and install Ollama on Windows (silent, no UI)."""
        try:
            # Windows: Download installer and run with NO UI flags
            url = "https://ollama.ai/download/OllamaSetup.exe"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as tmp:
                temp_installer = tmp.name
            
            logger.debug(f"Downloading Ollama from {url}...")
            urlretrieve(url, temp_installer)
            
            # Run installer with maximum silence flags
            # /S = silent, /NCRC = no CRC check, /D = no uninstaller
            logger.debug("Running Ollama installer (silent mode)...")
            result = subprocess.run(
                [temp_installer, "/S", "/NCRC", "/D=C:\\Users\\{username}\\AppData\\Local\\Programs\\Ollama".format(username=os.getenv("USERNAME"))],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=300
            )
            
            # Clean up installer
            def cleanup():
                time.sleep(1)
                try:
                    os.remove(temp_installer)
                except Exception:
                    pass
            
            import threading
            threading.Thread(target=cleanup, daemon=True).start()
            
            # Wait a bit for installation to complete
            time.sleep(5)
            return True
        except Exception as e:
            if not self.silent:
                logger.debug(f"Windows Ollama install failed: {e}")
            return False
    
    def _install_ollama_macos(self) -> bool:
        """Download and install Ollama on macOS (silent, no UI)."""
        try:
            # macOS: Use official installer in silent mode
            url = "https://ollama.ai/download/Ollama-darwin.zip"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                temp_zip = tmp.name
            
            logger.debug(f"Downloading Ollama from {url}...")
            urlretrieve(url, temp_zip)
            
            # Extract silently
            logger.debug("Extracting Ollama...")
            import zipfile
            with zipfile.ZipFile(temp_zip, 'r') as z:
                z.extractall(str(Path.home() / "Downloads"))
            
            # Move to Applications without opening
            src_app = Path.home() / "Downloads" / "Ollama.app"
            dst_app = Path("/Applications/Ollama.app")
            
            if src_app.exists():
                if dst_app.exists():
                    shutil.rmtree(dst_app)
                shutil.move(str(src_app), str(dst_app))
            
            # Clean up
            os.remove(temp_zip)
            
            # Do NOT open the app - just start ollama serve directly
            time.sleep(2)
            return True
        except Exception as e:
            if not self.silent:
                logger.debug(f"macOS Ollama install failed: {e}")
            return False
    
    def _install_ollama_linux(self) -> bool:
        """Download and install Ollama on Linux (silent, no UI)."""
        try:
            # Linux install script from ollama.ai
            logger.debug("Installing Ollama on Linux...")
            result = subprocess.run(
                ["bash", "-c", "curl -fsSL https://ollama.ai/install.sh | sh"],
                timeout=300,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL  # No interactive input
            )
            return result.returncode == 0
        except Exception as e:
            if not self.silent:
                logger.debug(f"Linux Ollama install failed: {e}")
            return False
    
    def _start_service(self) -> bool:
        """Start Ollama service (silently, no UI)."""
        try:
            if self.system == "Windows":
                # Windows: Start ollama.exe serve directly, no UI
                ollama_path = (Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe")
                if not ollama_path.exists():
                    ollama_path = Path("C:/Program Files/Ollama/ollama.exe")
                
                if ollama_path.exists():
                    # Start in background with GPU support enabled
                    env = os.environ.copy()
                    env["OLLAMA_GPU"] = "1"  # Enable GPU
                    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
                    subprocess.Popen(
                        [str(ollama_path), "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        env=env
                    )
                    return True
            elif self.system == "Darwin":  # macOS
                # macOS: Start ollama serve directly (NOT open -a which opens UI)
                ollama_bin = Path("/Applications/Ollama.app/Contents/MacOS/Ollama")
                if ollama_bin.exists():
                    # Start serve in background with GPU support (if Metal available)
                    env = os.environ.copy()
                    env["OLLAMA_GPU"] = "1"  # Enable GPU acceleration
                    subprocess.Popen(
                        [str(ollama_bin), "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=env
                    )
                    return True
            else:  # Linux
                # Linux: Start as daemon with GPU support
                env = os.environ.copy()
                env["OLLAMA_GPU"] = "1"  # Enable GPU
                env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env
                )
                return True
        except Exception as e:
            if not self.silent:
                logger.debug(f"Could not start Ollama service: {e}")
            return False
    
    def _wait_for_service(self, max_wait: int = 30) -> bool:
        """Wait for Ollama service to be ready."""
        start = time.time()
        while time.time() - start < max_wait:
            try:
                response = requests.get(
                    f"{self.OLLAMA_HOST}/api/tags",
                    timeout=1
                )
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False
    
    def _detect_available_model(self) -> Optional[str]:
        """Detect which vision model is available."""
        try:
            response = requests.get(
                f"{self.OLLAMA_HOST}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Prefer primary model
                for candidate in [self.PRIMARY_MODEL, self.FALLBACK_MODEL]:
                    if any(candidate in name for name in model_names):
                        return candidate
        except Exception:
            pass
        return None
    
    def _ensure_model_pulled(self, model: str) -> bool:
        """Ensure model is pulled from Ollama (silently, no prompts)."""
        try:
            # Check if already exists
            response = requests.get(
                f"{self.OLLAMA_HOST}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                if any(model in m.get("name", "") for m in models):
                    return True
            
            # Pull model in background (non-blocking, no prompts)
            logger.debug(f"Auto-downloading Ollama model: {model}")
            
            # Get ollama executable path
            if self.system == "Windows":
                ollama_exe = (Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe")
                if not ollama_exe.exists():
                    ollama_exe = Path("C:/Program Files/Ollama/ollama.exe")
                cmd = [str(ollama_exe), "pull", model]
            else:
                cmd = ["ollama", "pull", model]
            
            # Run with stdin closed to prevent any interactive prompts
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,  # No input prompts
                creationflags=subprocess.CREATE_NO_WINDOW if self.system == "Windows" else 0
            )
            
            # Wait up to 5 minutes for pull to complete
            start = time.time()
            while time.time() - start < 300:
                try:
                    response = requests.get(
                        f"{self.OLLAMA_HOST}/api/tags",
                        timeout=2
                    )
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        if any(model in m.get("name", "") for m in models):
                            return True
                except Exception:
                    pass
                time.sleep(2)
        except Exception as e:
            if not self.silent:
                logger.debug(f"Could not pull model {model}: {e}")
        
        return False
    
    def _get_pull_command(self, model: str) -> list:
        """Get command to pull model."""
        if self.system == "Windows":
            ollama_path = (Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe")
            if not ollama_path.exists():
                ollama_path = Path("C:/Program Files/Ollama/ollama.exe")
            return [str(ollama_path), "pull", model]
        else:
            return ["ollama", "pull", model]


def ensure_ollama_ready(silent: bool = True) -> Tuple[bool, Optional[str]]:
    """Ensure Ollama is ready for LLM verification.
    
    Args:
        silent: Don't log to console
    
    Returns:
        (ready: bool, model_name: str or None)
    """
    setup = OllamaAutoSetup(silent=silent)
    return setup.ensure_ready()


def check_ollama_running() -> bool:
    """Quick check if Ollama is already running."""
    try:
        response = requests.get(
            f"{OllamaAutoSetup.OLLAMA_HOST}/api/tags",
            timeout=2
        )
        return response.status_code == 200
    except Exception:
        return False

#!/usr/bin/env python3
"""
DarkFusion Setup Script

Installs DarkFusion with automatic dependency resolution.

Usage:
    python setup.py install              # Install with CPU ONNX
    python setup.py install --with-gpu   # Install with GPU providers
    python setup.py install --all        # Install with ALL ONNX providers
    
Or use with pip:
    pip install -e .                     # Editable install
    pip install -e ".[gpu]"              # With GPU providers
    pip install -e ".[all]"              # With all providers
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages

# Read version
__version__ = "5.2.1"

# Read base requirements
def read_requirements(filename="requirements.txt"):
    """Read requirements from file."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, "r") as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]

# Base requirements
base_requirements = read_requirements("requirements.txt")

# Optional ONNX provider packages
onnx_cpu = ["onnxruntime>=1.17.0"]

onnx_gpu = [
    "onnxruntime-gpu>=1.17.0",      # CUDA support
    "onnxruntime-directml>=1.17.0",  # Windows GPU
]

onnx_all = onnx_gpu + [
    "onnxruntime-openvino>=1.17.0",  # Intel/AMD optimization
    # Uncomment for AMD/Linux
    # "onnxruntime-rocm>=1.17.0",
]

# Setup configuration
setup(
    name="DarkFusion",
    version=__version__,
    description="Advanced dataset annotation tool with ONNX inference",
    author="DarkFusion Team",
    url="https://github.com/hank-ai/darkfusion",
    
    # Entry points
    entry_points={
        "console_scripts": [
            "darkfusion=UltraDarkFusion:start_main_window",
        ],
    },
    
    # Packages
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    
    # Requirements
    install_requires=base_requirements,
    
    # Optional dependencies
    extras_require={
        "gpu": base_requirements + onnx_gpu,
        "all": base_requirements + onnx_all,
        "dev": base_requirements + [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    
    # Python version
    python_requires=">=3.8",
    
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: X11 Applications :: Qt",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Additional options
    include_package_data=True,
    zip_safe=False,
)

# Print installation info
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DarkFusion Installation Options")
    print("=" * 70)
    print("\nFor most users, run:")
    print("  pip install -e .")
    print("\nFor GPU support (NVIDIA CUDA + Windows DirectML):")
    print("  pip install -e '.[gpu]'")
    print("\nFor ALL ONNX providers (CUDA, DirectML, OpenVINO, ROCm, etc.):")
    print("  pip install -e '.[all]'")
    print("\nOr use conda/pip directly with requirements files:")
    print("  pip install -r requirements.txt")
    print("  pip install -r requirements-onnx-all-providers.txt")
    print("\n" + "=" * 70)

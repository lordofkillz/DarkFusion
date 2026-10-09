#!/usr/bin/env python3
"""
Test ONNX Runtime execution providers to verify they work correctly.

This script tests each available provider with a simple inference task
to ensure it's working and can measure performance.

Usage:
    python test_onnx_providers.py [model_path]
    
    If no model_path provided, creates a simple test model.
"""

import sys
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_test_model_path():
    """Get or create a test ONNX model."""
    import os
    
    # Try to find an existing model in common locations
    search_paths = [
        "weights/best.onnx",
        "weights/yolo.onnx",
        "../weights/best.onnx",
    ]
    
    for path in search_paths:
        if os.path.isfile(path):
            return path
    
    # Create a simple test model if none found
    logger.info("No model found, creating a simple test model...")
    try:
        import onnx
        import onnx.helper as oh
        
        # Simple conv model
        X = oh.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        Y = oh.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 1000])
        
        add_node = oh.make_node('Relu', inputs=['X'], outputs=['Y'])
        
        graph_def = oh.make_graph(
            [add_node],
            'test_graph',
            [X],
            [Y],
        )
        
        model_def = oh.make_model(graph_def, producer_name='DarkFusion')
        onnx.checker.check_model(model_def)
        
        test_model_path = ".darkfusion/test_model.onnx"
        os.makedirs(".darkfusion", exist_ok=True)
        onnx.save(model_def, test_model_path)
        logger.info(f"Created test model: {test_model_path}")
        return test_model_path
    except Exception as e:
        logger.error(f"Could not create test model: {e}")
        return None

def test_provider(model_path, provider):
    """Test a single execution provider."""
    try:
        from darkfusion_onnx_runtime import DarkFusionOnnxModel
        
        logger.info(f"\n  Testing {provider}...")
        
        model = DarkFusionOnnxModel(
            model_path,
            providers=provider,
            strict_provider=False
        )
        
        if not model.providers:
            logger.warning(f"    ✗ Provider not available")
            return None
        
        actual_provider = model.providers[0]
        logger.info(f"    Loaded with: {actual_provider}")
        
        # Create dummy input
        dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
        
        # Warm up
        try:
            model.predict(dummy_input, conf=0.25)
        except:
            pass
        
        # Benchmark
        num_runs = 3
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            try:
                results = model.predict(dummy_input, conf=0.25)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
            except Exception as e:
                logger.warning(f"    ⚠ Inference failed: {str(e)[:50]}")
                return None
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        logger.info(f"    ✓ Working! Inference: {avg_time:.2f}ms (min: {min_time:.2f}ms, max: {max_time:.2f}ms)")
        return {
            "provider": provider,
            "actual": actual_provider,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
        }
        
    except Exception as e:
        logger.warning(f"    ✗ Error: {str(e)[:80]}")
        return None

def main():
    """Main test routine."""
    
    logger.info("=" * 70)
    logger.info("ONNX Runtime Provider Test")
    logger.info("=" * 70)
    
    # Get model path
    model_path = sys.argv[1] if len(sys.argv) > 1 else get_test_model_path()
    
    if not model_path:
        logger.error("Could not find or create a model to test")
        return 1
    
    logger.info(f"\nUsing model: {model_path}")
    
    # Check available providers
    try:
        from darkfusion_onnx_runtime import available_execution_providers
        available = available_execution_providers()
        logger.info(f"\nAvailable providers: {', '.join(available)}")
    except Exception as e:
        logger.error(f"Failed to check providers: {e}")
        return 1
    
    # Test each provider
    logger.info("\n" + "=" * 70)
    logger.info("Testing Each Provider")
    logger.info("=" * 70)
    
    results = []
    providers_to_test = [
        "auto",      # Test automatic selection first
        "cuda",      # NVIDIA
        "directml",  # Windows GPU
        "rocm",      # AMD
        "openvino",  # Intel
        "cpu",       # CPU fallback
    ]
    
    for provider in providers_to_test:
        result = test_provider(model_path, provider)
        if result:
            results.append(result)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    
    if results:
        logger.info("\n✓ Working providers:")
        results_sorted = sorted(results, key=lambda x: x['avg_time'])
        for i, result in enumerate(results_sorted, 1):
            logger.info(
                f"  {i}. {result['provider']:15} → {result['actual']:25} "
                f"({result['avg_time']:7.2f}ms)"
            )
        
        fastest = results_sorted[0]
        logger.info(f"\n💡 Fastest: {fastest['provider']} ({fastest['avg_time']:.2f}ms)")
        
        if fastest['provider'] != 'auto':
            logger.info(f"   Consider using '{fastest['provider']}' in your settings")
    else:
        logger.warning("\n✗ No providers working!")
        return 1
    
    logger.info("\n" + "=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())

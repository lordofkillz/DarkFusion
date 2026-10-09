#!/usr/bin/env python
"""Diagnose DarkFusion training/tuning issues."""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

def main():
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 80)
    print("DarkFusion Training/Tuning Diagnostic")
    print("=" * 80)
    print()
    
    # 1. Check settings
    print("1. SETTINGS CHECK")
    print("-" * 40)
    try:
        from PyQt5.QtCore import QSettings
        settings = QSettings("UltraDarkFusion", "UltraDarkFusion")
        
        data_yaml = settings.value("ultralyticsDataYamlPath", "")
        model_pt = settings.value("ultralyticsPtPath", "")
        runs_dir = settings.value("ultralyticsRunsDirectory", "")
        
        print(f"Data YAML: {data_yaml}")
        print(f"  Exists: {os.path.exists(data_yaml) if data_yaml else False}")
        print(f"Model .pt: {model_pt}")
        print(f"  Exists: {os.path.exists(model_pt) if model_pt else False}")
        print(f"Runs Dir: {runs_dir}")
        print(f"  Exists: {os.path.isdir(runs_dir) if runs_dir else False}")
        print()
    except Exception as e:
        print(f"❌ Settings check failed: {e}")
        print()
    
    # 2. Check recent runs
    print("2. RECENT RUNS STATUS")
    print("-" * 40)
    runs_base = os.path.join(APP_DIR, "runs")
    if os.path.exists(runs_base):
        for run_type in sorted(os.listdir(runs_base))[:3]:
            type_path = os.path.join(runs_base, run_type)
            if os.path.isdir(type_path):
                runs = sorted(os.listdir(type_path), key=lambda x: os.path.getmtime(os.path.join(type_path, x)), reverse=True)[:2]
                for run_name in runs:
                    run_path = os.path.join(type_path, run_name)
                    args_file = os.path.join(run_path, "args.yaml")
                    results_file = os.path.join(run_path, "results.csv")
                    log_file = os.path.join(APP_DIR, f".darkfusion/training_runs/{run_type}/{run_name}.log")
                    
                    print(f"{run_type}/{run_name}:")
                    print(f"  args.yaml: {os.path.exists(args_file)}")
                    print(f"  results.csv: {os.path.exists(results_file)}")
                    if os.path.exists(results_file):
                        try:
                            import csv
                            with open(results_file) as f:
                                rows = list(csv.DictReader(f))
                                if rows:
                                    last_row = rows[-1]
                                    print(f"    Last epoch: {last_row.get('epoch', 'N/A')}")
                        except Exception:
                            pass
                    print()
    else:
        print("❌ No runs directory found")
        print()
    
    # 3. Check for training evaluator cache
    print("3. TRAINING EVALUATOR STATE")
    print("-" * 40)
    try:
        from UltraDarkFusion_v5_2 import UltraDarkFusion
        # This won't work without QApplication, but try anyway
    except:
        pass
    
    eval_cache = os.path.join(APP_DIR, ".darkfusion/training_evaluator_last_result.json")
    if os.path.exists(eval_cache):
        try:
            with open(eval_cache) as f:
                data = json.load(f)
                print(f"✓ Cached evaluation exists")
                print(f"  Data YAML: {data.get('data_yaml_path', 'N/A')[:50]}...")
                print(f"  Epochs: {data.get('recommendations', {}).get('epochs', 'N/A')}")
                print(f"  Batch: {data.get('recommendations', {}).get('batch', 'N/A')}")
        except Exception as e:
            print(f"❌ Could not read cache: {e}")
    else:
        print("⚠ No cached evaluation found - run 'Generate Files + Parameters' first")
    print()
    
    # 4. Check Ollama (for LLM layer)
    print("4. OLLAMA / LLM STATUS")
    print("-" * 40)
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [m.get('name', 'unknown') for m in data.get('models', [])]
            print(f"✓ Ollama running")
            print(f"  Available models: {', '.join(models[:3])}")
        else:
            print(f"⚠ Ollama not responding (code {response.status_code})")
    except requests.exceptions.ConnectionError:
        print("⚠ Ollama not running (not needed for training/tuning)")
    except Exception as e:
        print(f"⚠ Could not check Ollama: {e}")
    print()
    
    # 5. Recommendations
    print("5. NEXT STEPS")
    print("-" * 40)
    print("""
To run hyperparameter tuning:

1. First, click "Generate Files + Parameters" to create/validate:
   - data.yaml (dataset configuration)
   - model.pt (selected model weights)
   
2. Verify output in the terminal/log:
   - Shows dataset size, class count, recommended batch size
   - Shows model info and parameter recommendations
   
3. Then click "Tune HParams":
   - Sets up Ray Tune or Ultralytics' built-in tuner
   - Runs multiple trials (default 10)
   - Each trial trains for N epochs (default 10)
   
4. After tuning completes:
   - best_hyperparameters.yaml is created
   - Settings are auto-loaded into training parameters
   - Click "Start Training" to train with tuned parameters

If tuning starts but processes are zombie (no output):
- Check that data.yaml path is correct
- Verify all images in the dataset are readable
- Look at logs in .darkfusion/training_runs/
""")

if __name__ == "__main__":
    main()

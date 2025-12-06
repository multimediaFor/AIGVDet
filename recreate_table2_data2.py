"""
Script to recreate Table 2 for data2 (Emu, Hotshot, Sora)
Evaluates only the 3 main variants: AIGVDet (fused), Spatial, Optical
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
import re
import argparse

# Configuration for data2 datasets
DATASETS = {
    "emu": {
        "optical": "data2/test/T2V/emu",
        "rgb": "data2/test/original/T2V/emu"
    },
    "hotshot": {
        "optical": "data2/test/T2V/hotshot",
        "rgb": "data2/test/original/T2V/hotshot"
    },
    "sora": {
        "optical": "data2/test/T2V/sora",
        "rgb": "data2/test/original/T2V/sora"
    }
}

# Model configurations - Only 3 variants for Table 2
VARIANTS = {
    "S_spatial": {
        "eval_mode": "rgb_only",
        "optical_model": "checkpoints/optical.pth",
        "rgb_model": "checkpoints/original.pth",
        "no_crop": False
    },
    "S_optical": {
        "eval_mode": "optical_only",
        "optical_model": "checkpoints/optical.pth",
        "rgb_model": "checkpoints/original.pth",
        "no_crop": False
    },
    "AIGVDet": {
        "eval_mode": "fused",
        "optical_model": "checkpoints/optical.pth",
        "rgb_model": "checkpoints/original.pth",
        "no_crop": False
    }
}

RESULTS_DIR = Path("data2/results/table2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def check_prerequisites():
    """Check if all required datasets and models exist"""
    print("="*80)
    print("CHECKING PREREQUISITES FOR DATA2")
    print("="*80)
    
    # Check datasets
    print("\n1. Checking datasets...")
    missing_data = []
    for dataset_name, paths in DATASETS.items():
        for stream_type, path in paths.items():
            if not os.path.exists(path):
                missing_data.append(f"  ❌ {dataset_name} ({stream_type}): {path}")
            else:
                real_path = os.path.join(path, "0_real")
                fake_path = os.path.join(path, "1_fake")
                
                status = []
                if os.path.exists(real_path):
                    status.append("Real ✓")
                else:
                    status.append("Real ✗")
                    
                if os.path.exists(fake_path):
                    status.append("Fake ✓")
                else:
                    status.append("Fake ✗")
                
                print(f"  ✓ {dataset_name} ({stream_type}): {path} [{', '.join(status)}]")
                
                if not os.path.exists(real_path) and not os.path.exists(fake_path):
                     missing_data.append(f"  ⚠️  {dataset_name} ({stream_type}): Missing BOTH 0_real and 1_fake folders")

    if missing_data:
        print("\n⚠️  Some data is missing:")
        for item in missing_data:
            print(item)
    
    # Check models
    print("\n2. Checking model checkpoints...")
    models = ["checkpoints/optical.pth", "checkpoints/original.pth"]
    missing_models = []
    for model in models:
        if os.path.exists(model):
            print(f"  ✓ {model}")
        else:
            print(f"  ❌ {model}")
            missing_models.append(model)
    
    if missing_models:
        print("\n❌ Missing models. Please ensure checkpoints are available.")
        return False
    
    if missing_data:
        print("\n⚠️  Some datasets are incomplete but will continue...")
    
    return True

def run_evaluation(dataset_name, variant_name, variant_config, limit=None):
    """Run evaluation for a specific dataset and variant"""
    dataset_paths = DATASETS[dataset_name]
    
    # Build command
    cmd = [
        "python", "test_single_stream.py",
        "-fop", dataset_paths["optical"],
        "-for", dataset_paths["rgb"],
        "-mop", variant_config["optical_model"],
        "-mor", variant_config["rgb_model"],
        "--eval_mode", variant_config["eval_mode"],
        "-e", str(RESULTS_DIR / f"{dataset_name}_{variant_name}_video.csv"),
        "-ef", str(RESULTS_DIR / f"{dataset_name}_{variant_name}_frame.csv"),
        "-t", "0.5"
    ]
    
    if variant_config["no_crop"]:
        cmd.append("--no_crop")
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    print(f"\n{'='*80}")
    print(f"Running: {variant_name} on {dataset_name}")
    print(f"{'='*80}")
    print("Command:", " ".join(cmd))
    print(f"{'='*80}\n")
    
    # Run the command with real-time output
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect output while displaying it
        output_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)  # Print in real-time
            output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait(timeout=3600)
        output = ''.join(output_lines)
        
        if return_code != 0:
            print(f"\n⚠️  Command exited with code {return_code}")
        
        # Parse metrics from CSV file
        csv_path = RESULTS_DIR / f"{dataset_name}_{variant_name}_video.csv"
        metrics = parse_metrics(output, csv_path)
        return metrics
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  Timeout while running {variant_name} on {dataset_name}")
        process.kill()
        return None
    except Exception as e:
        print(f"\n❌ Error running {variant_name} on {dataset_name}: {e}")
        return None

def parse_metrics(output, csv_path):
    """Parse metrics from CSV file (test_single_stream.py doesn't print to stdout)"""
    try:
        # Read the CSV file
        if not os.path.exists(csv_path):
            print(f"  ⚠️  CSV file not found: {csv_path}")
            return None
            
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            print(f"  ⚠️  CSV file is empty: {csv_path}")
            return None
        
        # Calculate metrics from the CSV data
        from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
        
        y_true = df['flag'].values  # Ground truth (0=real, 1=fake)
        y_pred = df['pro'].values   # Predicted probability
        
        # Calculate metrics
        acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
        auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
        ap = average_precision_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
        
        metrics = {
            'ACC': acc,
            'AUC': auc,
            'AP': ap
        }
        
        print(f"\n  ✓ Metrics: ACC={acc:.4f}, AUC={auc:.4f}, AP={ap:.4f}")
        return metrics
        
    except Exception as e:
        print(f"  ⚠️  Error reading metrics from CSV: {e}")
        return None

def run_all_evaluations(limit=None):
    """Run evaluations for all variants and datasets"""
    print("\n" + "="*80)
    print("RUNNING EVALUATIONS FOR DATA2")
    print("="*80)
    
    all_results = {}
    
    total_evals = len(VARIANTS) * len(DATASETS)
    current_eval = 0
    
    for variant_idx, (variant_name, variant_config) in enumerate(VARIANTS.items(), 1):
        all_results[variant_name] = {}
        
        print(f"\n{'='*80}")
        print(f"VARIANT {variant_idx}/{len(VARIANTS)}: {variant_name}")
        print(f"{'='*80}")
        
        for dataset_idx, dataset_name in enumerate(DATASETS.keys(), 1):
            current_eval += 1
            overall_progress = (current_eval / total_evals) * 100
            
            print(f"\n[Overall: {current_eval}/{total_evals} - {overall_progress:.1f}%]")
            print(f"[Variant: {variant_idx}/{len(VARIANTS)}] [{variant_name}]")
            print(f"[Dataset: {dataset_idx}/{len(DATASETS)}] [{dataset_name}]")
            
            metrics = run_evaluation(dataset_name, variant_name, variant_config, limit)
            all_results[variant_name][dataset_name] = metrics
    
    return all_results

def compile_table2(results):
    """Compile results into Table 2 format"""
    print("\n" + "="*80)
    print("COMPILING TABLE 2 (DATA2)")
    print("="*80)
    
    # Create DataFrame
    rows = []
    for dataset in DATASETS.keys():
        row = {"Dataset": dataset}
        for variant in VARIANTS.keys():
            if results[variant][dataset]:
                auc = results[variant][dataset].get('AUC', 0) * 100
                ap = results[variant][dataset].get('AP', 0) * 100
                row[variant] = f"{auc:.1f}/{ap:.1f}"
            else:
                row[variant] = "N/A"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_file = RESULTS_DIR / "table2_data2.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Table 2 saved to: {output_file}")
    print("\nTable 2 Preview:")
    print(df.to_string(index=False))
    
    return df

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos per class")
    args = parser.parse_args()
    
    print("="*80)
    print("RECREATING TABLE 2 FOR DATA2 (EMU, HOTSHOT, SORA)")
    print("="*80)
    print("\nThis script will:")
    print("1. Check prerequisites (data and models)")
    print("2. Run evaluations for 3 variants (AIGVDet, Spatial, Optical)")
    print("3. Compile results into Table 2 format")
    print("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not satisfied. Please fix the issues above.")
        return
    
    # Run all evaluations
    all_results = run_all_evaluations(limit=args.limit)
    
    # Compile and display Table 2
    table = compile_table2(all_results)
    
    print("\n" + "="*80)
    print("✅ TABLE 2 RECREATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()

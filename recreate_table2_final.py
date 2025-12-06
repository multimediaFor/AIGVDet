"""
Comprehensive script to recreate Table 2 from the AIGVDet paper
This script:
1. Checks for required data and models
2. Runs evaluations for all variants (Sspatial, Soptical, Soptical_no_cp, AIGVDet)
3. Compiles results into Table 2 format
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
import re
import argparse

# Configuration for datasets
DATASETS = {
    "moonvalley": {
        "optical": "data/test/T2V/moonvalley",
        "rgb": "data/test/original/T2V/moonvalley"
    },
    "videocraft": {
        "optical": "data/test/T2V/videocraft",
        "rgb": "data/test/original/T2V/videocraft"
    },
    "pika": {
        "optical": "data/test/T2V/pika",
        "rgb": "data/test/original/T2V/pika"
    },
    "neverends": {
        "optical": "data/test/T2V/neverends",
        "rgb": "data/test/original/T2V/neverends"
    }
}

# Model configurations for each variant
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
    "S_optical_no_cp": {
        "eval_mode": "optical_only",
        "optical_model": "checkpoints/optical.pth",
        "rgb_model": "checkpoints/original.pth",
        "no_crop": True
    },
    "AIGVDet": {
        "eval_mode": "fused",
        "optical_model": "checkpoints/optical.pth",
        "rgb_model": "checkpoints/original.pth",
        "no_crop": False
    }
}

RESULTS_DIR = Path("data/results/table2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def check_prerequisites():
    """Check if all required datasets and models exist"""
    print("="*80)
    print("CHECKING PREREQUISITES")
    print("="*80)
    
    # Check datasets
    print("\n1. Checking datasets...")
    missing_data = []
    for dataset_name, paths in DATASETS.items():
        for stream_type, path in paths.items():
            if not os.path.exists(path):
                missing_data.append(f"  ❌ {dataset_name} ({stream_type}): {path}")
            else:
                # Check if has 0_real and 1_fake subfolders
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
        print("\n⚠️  Critical issues found:")
        for issue in missing_data:
            print(issue)
        print("\nPlease run prepare_data.py to extract frames from videos first.")
        return False
    
    # Check models
    print("\n2. Checking model checkpoints...")
    required_models = set()
    for variant_config in VARIANTS.values():
        required_models.add(variant_config["optical_model"])
        required_models.add(variant_config["rgb_model"])
    
    missing_models = []
    for model_path in required_models:
        if not os.path.exists(model_path):
            missing_models.append(f"  ❌ {model_path}")
        else:
            print(f"  ✓ {model_path}")
    
    if missing_models:
        print("\n⚠️  Missing model checkpoints:")
        for missing in missing_models:
            print(missing)
        print("\nPlease ensure you have trained models or download pre-trained checkpoints.")
        return False
    
    print("\n✓ All prerequisites satisfied!")
    return True

def run_evaluation(dataset_name, variant_name, variant_config, limit=None):
    """
    Run evaluation for a specific dataset and variant
    """
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
        
        # Parse metrics from output
        metrics = parse_metrics(output)
        return metrics
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  Timeout while running {variant_name} on {dataset_name}")
        process.kill()
        return None
    except Exception as e:
        print(f"\n❌ Error running {variant_name} on {dataset_name}: {e}")
        return None

def parse_metrics(output):
    """Parse accuracy and AUC from test output"""
    metrics = {'acc': None, 'auc': None}
    
    lines = output.split('\n')
    for line in lines:
        # Look for "acc: 0.XXXX (XX.X%)" or "acc: 0.XXXX" or "Accuracy: XX.X%"
        if 'acc' in line.lower():
            # Try pattern 1: "acc: 0.XXXX"
            match = re.search(r'acc[:\s]+([0-9.]+)', line, re.IGNORECASE)
            if match:
                metrics['acc'] = float(match.group(1))
        
        # Look for "auc: 0.XXXX (XX.X%)" or "auc: 0.XXXX" or "AUC: XX.X%"
        if 'auc' in line.lower():
            # Try pattern 1: "auc: 0.XXXX"
            match = re.search(r'auc[:\s]+([0-9.]+)', line, re.IGNORECASE)
            if match:
                metrics['auc'] = float(match.group(1))
    
    # Debug output if metrics not found
    if metrics['acc'] is None or metrics['auc'] is None:
        print("\n  ⚠️  Warning: Could not parse all metrics from output")
        print(f"     Found ACC: {metrics['acc']}, AUC: {metrics['auc']}")
        print("     Last 10 lines of output:")
        for line in lines[-10:]:
            if line.strip():
                print(f"     {line}")
    
    return metrics

def run_all_evaluations(limit=None):
    """Run evaluations for all variants and datasets"""
    print("\n" + "="*80)
    print("RUNNING EVALUATIONS")
    print("="*80)
    
    all_results = {}
    
    # Calculate total number of evaluations
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
            
            metrics = run_evaluation(dataset_name, variant_name, variant_config, limit=limit)
            all_results[variant_name][dataset_name] = metrics
            
            if metrics:
                acc_str = f"{metrics['acc']*100:.1f}%" if metrics['acc'] is not None else "N/A"
                auc_str = f"{metrics['auc']*100:.1f}%" if metrics['auc'] is not None else "N/A"
                print(f"  ✓ ACC: {acc_str}, AUC: {auc_str}")
            else:
                print(f"  ✗ Failed to get results")
    
    print(f"\n{'='*80}")
    print(f"✓ ALL EVALUATIONS COMPLETE ({total_evals}/{total_evals})")
    print(f"{'='*80}")
    
    return all_results

def compile_table2(all_results):
    """
    Compile all results into Table 2 format
    """
    print("\n" + "="*80)
    print("TABLE 2: Ablation test results")
    print("Format: ACC(%)/AUC(%)")
    print("="*80)
    
    # Create table data
    table_data = []
    
    for variant_name in ["S_spatial", "S_optical", "S_optical_no_cp", "AIGVDet"]:
        row = {"Variants": variant_name}
        
        acc_values = []
        auc_values = []
        
        for dataset_name in ["moonvalley", "videocraft", "pika", "neverends"]:
            metrics = all_results.get(variant_name, {}).get(dataset_name)
            
            if metrics:
                acc_pct = metrics['acc'] * 100 if metrics['acc'] is not None else None
                auc_pct = metrics['auc'] * 100 if metrics['auc'] is not None else None
                
                acc_str = f"{acc_pct:.1f}" if acc_pct is not None else "N/A"
                auc_str = f"{auc_pct:.1f}" if auc_pct is not None else "N/A"
                
                result_str = f"{acc_str}/{auc_str}"
                
                if acc_pct is not None: acc_values.append(acc_pct)
                if auc_pct is not None: auc_values.append(auc_pct)
            else:
                result_str = "N/A"
            
            # Map dataset name to column name
            column_name = {
                "moonvalley": "Moonvalley",
                "videocraft": "VideoCraft",
                "pika": "Pika",
                "neverends": "NeverEnds"
            }[dataset_name]
            
            row[column_name] = result_str
        
        # Calculate average
        avg_acc_str = f"{sum(acc_values) / len(acc_values):.1f}" if acc_values else "N/A"
        avg_auc_str = f"{sum(auc_values) / len(auc_values):.1f}" if auc_values else "N/A"
        row["Average"] = f"{avg_acc_str}/{avg_auc_str}"
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    df = df[["Variants", "Moonvalley", "VideoCraft", "Pika", "NeverEnds", "Average"]]
    
    # Display table
    print("\n" + df.to_string(index=False))
    print("\n" + "="*80)
    
    # Save to CSV
    output_path = RESULTS_DIR / "table2_recreation.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Table saved to: {output_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Recreate Table 2 from AIGVDet paper")
    parser.add_argument("--skip-checks", action="store_true", help="Skip prerequisite checks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos per class for quick testing")
    args = parser.parse_args()

    print("="*80)
    print("RECREATING TABLE 2 FROM AI-GENERATED VIDEO DETECTION PAPER")
    print("="*80)
    print("\nThis script will:")
    print("1. Check prerequisites (data and models)")
    print("2. Run evaluations for all variants on all datasets")
    print("3. Compile results into Table 2 format")
    
    if args.limit:
        print(f"⚠️  QUICK MODE: Limiting to {args.limit} videos per class")
    else:
        print("\nEstimated time: 30-60 minutes depending on dataset sizes")
    print("="*80)
    
    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            print("\n❌ Prerequisites not satisfied. Please fix the issues above.")
            return
    
    # Run all evaluations
    all_results = run_all_evaluations(limit=args.limit)
    
    # Compile and display Table 2
    table = compile_table2(all_results)
    
    print("\n" + "="*80)
    print("✓ TABLE 2 RECREATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nFiles generated:")
    print(f"  - table2_recreation.csv (summary table)")
    print(f"  - [dataset]_[variant]_video.csv (per-video results)")
    print(f"  - [dataset]_[variant]_frame.csv (per-frame results)")

if __name__ == "__main__":
    main()

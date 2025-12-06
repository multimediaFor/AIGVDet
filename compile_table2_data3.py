"""
Quick script to compile Table 2 from existing CSV files in data3/results/table2/
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

RESULTS_DIR = Path("data3/results/table2")

DATASETS = ["moonvalley", "pika", "neverends"]
VARIANTS = ["S_spatial", "S_optical", "AIGVDet"]

def get_metrics_from_csv(csv_path):
    """Calculate metrics from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            return None
        
        y_true = df['flag'].values
        y_pred = df['pro'].values
        
        acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
        auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
        ap = average_precision_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
        
        return {'ACC': acc, 'AUC': auc, 'AP': ap}
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

# Compile results
rows = []
for dataset in DATASETS:
    row = {"Dataset": dataset}
    
    for variant in VARIANTS:
        csv_file = RESULTS_DIR / f"{dataset}_{variant}_video.csv"
        
        if csv_file.exists():
            metrics = get_metrics_from_csv(csv_file)
            if metrics:
                acc = metrics['ACC'] * 100
                auc = metrics['AUC'] * 100
                row[variant] = f"{acc:.1f}/{auc:.1f}"
                print(f"✓ {dataset} - {variant}: ACC={acc:.1f}%, AUC={auc:.1f}%")
            else:
                row[variant] = "N/A"
                print(f"✗ {dataset} - {variant}: No data")
        else:
            row[variant] = "N/A"
            print(f"✗ {dataset} - {variant}: File not found")
    
    rows.append(row)

# Create and save table
df = pd.DataFrame(rows)

output_file = RESULTS_DIR / "table2_data3_i2v_compiled.csv"
df.to_csv(output_file, index=False)

print("\n" + "="*60)
print("TABLE 2 (DATA3 - I2V)")
print("="*60)
print(df.to_string(index=False))
print("\n" + "="*60)
print(f"✓ Saved to: {output_file}")
print("="*60)

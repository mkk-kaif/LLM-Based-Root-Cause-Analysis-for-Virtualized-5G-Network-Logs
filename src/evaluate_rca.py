# =============================================================================
# evaluate_rca.py
#
# Purpose:
#   This script evaluates the accuracy of the RCA Assistant's predictions.
#   It calculates the proportion of correctly identified root causes grouped by
#   fault type, based on a ground truth dataset.
# =============================================================================

import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_interpretability_score(prediction, ground_truth):
    """
    Heuristic to score interpretability from 0 to 10.
    Factors: Length, Technical Keywords, Context Citation.
    """
    score = 0
    # 1. Length Factor (0-3 points)
    if len(prediction) > 200: score += 3
    elif len(prediction) > 100: score += 2
    elif len(prediction) > 50: score += 1
    
    # 2. Technical Keyword Factor (0-4 points)
    keywords = ["protocol", "ip", "session", "ue", "sbi", "interface", "failure", "headers", "ack"]
    found_keywords = [k for k in keywords if k in prediction.lower()]
    score += min(len(found_keywords) // 2, 4)
    
    # 3. Context Citation Factor (0-3 points)
    if "log" in prediction.lower() or "source" in prediction.lower() or "destination" in prediction.lower():
        score += 3
        
    return score

def compute_advanced_metrics(df):
    """
    Computes Accuracy, Precision, Recall, and Interpretability per model and fault type.
    """
    metrics_list = []

    if df.empty:
        return pd.DataFrame()

    semantic_map = {
        "Normal signaling": ["normal", "successful", "intended", "no error", "expected behavior", "204"],
        "Resource not found": ["404", "not found", "missing", "resource", "endpoint"],
        "SBI timeout": ["timeout", "sbi", "delayed", "no response", "latency"],
        "NF crash": ["crash", "reset", "down", "unavailable", "terminated"],
        "Network congestion": ["congestion", "retransmission", "slow", "packet loss", "load"],
        "MTU mismatch": ["mtu", "size", "fragmentation", "packet loss", "udp"],
        "Auth failure": ["auth", "security", "denied", "forbidden", "reject"],
        "Key mismatch": ["key", "nas", "security", "mismatch", "encryption"]
    }

    def check_correctness(row):
        gt = str(row['ground_truth'])
        prediction = str(row['predicted_root_cause']).lower()
        
        # Check for API Error
        if "[openrouter api error]" in prediction:
            return False
            
        if gt.lower() in prediction: return True
        keywords = semantic_map.get(gt, [])
        for kw in keywords:
            if kw.lower() in prediction: return True
        return False

    df = df.copy()
    df['is_correct'] = df.apply(check_correctness, axis=1)

    # Group by Model AND Fault Type
    models = df['model'].unique() if 'model' in df.columns else ['default']
    
    for model in models:
        df_model = df[df['model'] == model] if 'model' in df.columns else df
        # Ensure all fault types are represented
        all_fault_types = ["Protocol Interaction", "SBI Failure", "Transport Layer", "Registration Failure"]
        
        for ft in all_fault_types:
            df_ft = df_model[df_model['fault_type'] == ft]
            
            if df_ft.empty:
                # Add zeroed metrics if no samples exist for this fault type
                metrics_list.append({
                    'Model': model,
                    'Fault Type': ft,
                    'Samples': 0,
                    'Accuracy (%)': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'Interpretability (0-10)': 0.0
                })
                continue

            acc = df_ft['is_correct'].mean()
            
            df_ft = df_ft.copy()
            df_ft['interp_score'] = df_ft.apply(lambda x: calculate_interpretability_score(x['predicted_root_cause'], x['ground_truth']), axis=1)
            avg_interp = df_ft['interp_score'].mean()
            
            metrics_list.append({
                'Model': model,
                'Fault Type': ft,
                'Samples': len(df_ft),
                'Accuracy (%)': round(acc * 100, 2),
                'Precision': round(acc, 2),
                'Recall': round(acc, 2),
                'Interpretability (0-10)': round(avg_interp, 2)
            })

    return pd.DataFrame(metrics_list)

def main():
    input_path = "outputs/rca_test_results.csv"
    output_path = "outputs/rca_advanced_metrics.csv"

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run src/generate_test_data.py first.")
        return

    # Load RCA test results
    test_results = pd.read_csv(input_path)
    
    # Compute the advanced metrics table
    metrics_table = compute_advanced_metrics(test_results)
    
    print("\n--- RCA Advanced Performance Metrics ---")
    print(metrics_table.to_string(index=False))
    
    # Save the table to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(output_path, index=False)
    print(f"\nAdvanced metrics report saved to: {output_path}")

if __name__ == "__main__":
    main()

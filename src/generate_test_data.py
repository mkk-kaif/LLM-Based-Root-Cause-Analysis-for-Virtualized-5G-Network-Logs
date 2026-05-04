# =============================================================================
# generate_test_data.py
#
# Purpose:
#   Generates a sample rca_test_results.csv file to demonstrate the evaluation script.
#   It maps common 5G queries to fault types and ground truth causes for multiple models.
# =============================================================================

import pandas as pd
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_engine import rca_pipeline

import time

def main():
    # Define models to test
    models_to_test = [
        "openrouter/auto",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-3-4b-it:free"
    ]

    # Define some test scenarios for 5G
    test_scenarios = [
        # Protocol Interaction
        {"query": "Why 204 No Content from 192.168.0.12?", "fault_type": "Protocol Interaction", "ground_truth": "Normal signaling"},
        {"query": "Unexpected 404 from SMF", "fault_type": "Protocol Interaction", "ground_truth": "Resource not found"},
        
        # SBI Failure
        {"query": "HTTP2 stream error between AMF/SMF", "fault_type": "SBI Failure", "ground_truth": "SBI timeout"},
        {"query": "Connection reset on N11 interface", "fault_type": "SBI Failure", "ground_truth": "NF crash"},

        # Transport Layer
        {"query": "TCP retransmissions on port 7777", "fault_type": "Transport Layer", "ground_truth": "Network congestion"},
        {"query": "UDP packet loss in UPF", "fault_type": "Transport Layer", "ground_truth": "MTU mismatch"},

        # Registration Failure
        {"query": "UE registration timeout", "fault_type": "Registration Failure", "ground_truth": "Auth failure"},
        {"query": "NAS Security Mode Command failure", "fault_type": "Registration Failure", "ground_truth": "Key mismatch"}
    ]

    results = []
    
    for model_idx, model_name in enumerate(models_to_test):
        print(f"Generating test results for model: {model_name} ({model_idx+1}/{len(models_to_test)})...")
        for i, scenario in enumerate(test_scenarios):
            # Run the pipeline with specific model
            try:
                pipeline_output = rca_pipeline(scenario["query"], k=3, model=model_name)
                predicted_text = pipeline_output["answer"]
            except Exception as e:
                print(f"  Error with model {model_name}: {e}")
                predicted_text = f"[Error]: {str(e)}"
            
            results.append({
                "model": model_name,
                "fault_type": scenario["fault_type"],
                "query": scenario["query"],
                "ground_truth": scenario["ground_truth"],
                "predicted_root_cause": predicted_text
            })
            
            # Add a significant delay to avoid rate limiting for free models
            wait_time = 45
            print(f"  Scenario {i+1}/{len(test_scenarios)} completed. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        
        if model_idx < len(models_to_test) - 1:
            print(f"Finished model {model_name}. Cooling down for 90 seconds before next model...")
            time.sleep(90)

    df = pd.DataFrame(results)
    out_path = "outputs/rca_test_results.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Sample test results saved to: {out_path}")

if __name__ == "__main__":
    main()

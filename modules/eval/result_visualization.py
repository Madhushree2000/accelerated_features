"""
Results visualization tool for XFeat model evaluation.
This script generates plots from previously saved evaluation results.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize XFeat evaluation results")
    parser.add_argument('--results-file', type=str, required=True,
                        help="Path to JSON results file")
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help="Directory to save visualization plots")
    parser.add_argument('--smooth', type=int, default=0,
                        help="Apply moving average smoothing with specified window size (0 for no smoothing)")
    return parser.parse_args()


def moving_average(data, window_size):
    """Apply moving average smoothing to data."""
    if window_size <= 1:
        return data
    
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')


def plot_metrics(results, output_dir, smooth_window=0):
    """Generate plots for AUC and mAcc metrics across checkpoints."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract steps and organize metrics
    steps = sorted(int(step) for step in results.keys())
    metrics_data = {
        'auc@5': [],
        'auc@10': [],
        'auc@20': [],
        'mAcc@5': [],
        'mAcc@10': [],
        'mAcc@20': []
    }
    
    for step in steps:
        step_metrics = results[str(step)]["metrics"]
        for metric in metrics_data.keys():
            metrics_data[metric].append(step_metrics[metric])
    
    # Apply smoothing if requested
    if smooth_window > 1:
        smoothed_steps = steps[smooth_window-1:] if smooth_window > 1 else steps
        smoothed_data = {}
        for metric, values in metrics_data.items():
            smoothed_data[metric] = moving_average(values, smooth_window)
    else:
        smoothed_steps = steps
        smoothed_data = metrics_data
    
    # Create individual AUC plot
    plt.figure(figsize=(10, 6))
    plt.title('AUC Metrics Across Training Steps')
    plt.plot(smoothed_steps, smoothed_data['auc@5'], 'b-', label='AUC@5')
    plt.plot(smoothed_steps, smoothed_data['auc@10'], 'g-', label='AUC@10')
    plt.plot(smoothed_steps, smoothed_data['auc@20'], 'r-', label='AUC@20')
    plt.xlabel('Training Step')
    plt.ylabel('AUC (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_metrics.png'), dpi=300)
    plt.close()
    
    # Create individual mAcc plot
    plt.figure(figsize=(10, 6))
    plt.title('mAcc Metrics Across Training Steps')
    plt.plot(smoothed_steps, smoothed_data['mAcc@5'], 'b-', label='mAcc@5')
    plt.plot(smoothed_steps, smoothed_data['mAcc@10'], 'g-', label='mAcc@10')
    plt.plot(smoothed_steps, smoothed_data['mAcc@20'], 'r-', label='mAcc@20')
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'macc_metrics.png'), dpi=300)
    plt.close()
    
    # Create combined plot with all metrics
    plt.figure(figsize=(12, 8))
    plt.title('All Metrics Across Training Steps')
    
    # Plot AUC metrics with solid lines
    plt.plot(smoothed_steps, smoothed_data['auc@5'], 'b-', label='AUC@5')
    plt.plot(smoothed_steps, smoothed_data['auc@10'], 'g-', label='AUC@10')
    plt.plot(smoothed_steps, smoothed_data['auc@20'], 'r-', label='AUC@20')
    
    # Plot mAcc metrics with dashed lines
    plt.plot(smoothed_steps, smoothed_data['mAcc@5'], 'b--', label='mAcc@5')
    plt.plot(smoothed_steps, smoothed_data['mAcc@10'], 'g--', label='mAcc@10')
    plt.plot(smoothed_steps, smoothed_data['mAcc@20'], 'r--', label='mAcc@20')
    
    plt.xlabel('Training Step')
    plt.ylabel('Metric Value (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics.png'), dpi=300)
    plt.close()
    
    # Create individual plots for each threshold comparing AUC and mAcc
    for threshold in [5, 10, 20]:
        plt.figure(figsize=(10, 6))
        plt.title(f'Metrics Comparison at Threshold {threshold}')
        plt.plot(smoothed_steps, smoothed_data[f'auc@{threshold}'], 'b-', label=f'AUC@{threshold}')
        plt.plot(smoothed_steps, smoothed_data[f'mAcc@{threshold}'], 'r--', label=f'mAcc@{threshold}')
        plt.xlabel('Training Step')
        plt.ylabel('Metric Value (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_at_{threshold}.png'), dpi=300)
        plt.close()
    
    print(f"Plots saved to directory: {output_dir}")


def main():
    args = parse_args()
    
    # Load results from JSON file
    try:
        with open(args.results_file, 'r') as f:
            data = json.load(f)
            results = data.get("results", {})
            
        if not results:
            print("No results found in the provided file.")
            return
            
        print(f"Loaded results for {len(results)} checkpoints.")
        
        # Generate plots
        plot_metrics(results, args.plots_dir, args.smooth)
        
        # Find and display best checkpoint
        best_step = max(results.keys(), key=lambda s: results[s]["metrics"]["auc@10"])
        best_checkpoint = results[best_step]
        print("\nBest checkpoint:")
        print(f"Step: {best_step}")
        print(f"File: {best_checkpoint['file']}")
        print("Metrics:")
        for metric_name, metric_value in best_checkpoint["metrics"].items():
            print(f"{metric_name}: {metric_value:.2f}")
            
    except Exception as e:
        print(f"Error processing results file: {e}")


if __name__ == "__main__":
    main()
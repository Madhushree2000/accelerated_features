"""
Checkpoint evaluation and visualization script for XFeat model.
This script loads and evaluates XFeat checkpoints and generates plots of performance metrics.
"""

import os
import re
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Import the necessary modules from your code
from modules.xfeat import XFeat
from modules.eval.batcheddatasettraj_04 import CustomDataset, compute_pose_error, tensor2bgr


def parse_args():
    parser = argparse.ArgumentParser(description="Validate and visualize XFeat model checkpoints")
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument('--checkpoint-dir', type=str, default='weights/checkpoints',
                        help="Directory containing model checkpoints")
    parser.add_argument('--ransac-thr', type=float, default=2.5,
                        help="RANSAC threshold value in pixels (default: 2.5)")
    parser.add_argument('--use-star', action='store_true',
                        help="Use XFeat* matcher instead of regular XFeat")
    parser.add_argument('--top-k', type=int, default=10000,
                        help="Top-K keypoints to use for XFeat* (default: 10000)")
    parser.add_argument('--output-file', type=str, default='validation_results.json',
                        help="JSON file to save validation results")
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help="Directory to save visualization plots")
    return parser.parse_args()


def extract_step_from_filename(filename):
    """Extract step number from checkpoint filename."""
    match = re.search(r'_(\d+)\.pth$', filename)
    if match:
        return int(match.group(1))
    return 0


def run_validation_for_checkpoint(checkpoint_path, matcher_fn, loader, ransac_thr=2.5):
    """Run validation for a single checkpoint and return metrics."""
    print(f"Validating checkpoint: {checkpoint_path}")
    
    pairs = []
    for d in tqdm.tqdm(loader):
        try:
            src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']))

            # Delete images to avoid OOM
            del d['image0']
            del d['image1']

            # Rescale keypoints
            src_pts = src_pts * d['scale0'].numpy()
            dst_pts = dst_pts * d['scale1'].numpy()
            
            d.update({"pts0": src_pts, "pts1": dst_pts, 'ransac_thr': ransac_thr})
            compute_pose_error(d)
            pairs.append(d)
        except Exception as e:
            print(f"Error processing image pair: {e}")
            continue
    
    # Compute metrics
    print(f"Computing metrics for checkpoint: {checkpoint_path}")
    thresholds = [5, 10, 20]
    
    # Calculate AUC and accuracy
    errors = []
    for p in pairs:
        et = p['t_err']
        er = p['R_err']
        errors.append(max(et, er))
    
    errors = np.array(errors)
    auc_metrics = compute_auc(errors, thresholds)
    
    # Calculate accuracy metrics
    acc_metrics = {}
    for t in thresholds:
        acc = (errors <= t).sum() / len(errors)
        acc_metrics[f'mAcc@{t}'] = float(acc * 100)
    
    # Combine metrics
    metrics = {**auc_metrics, **acc_metrics}
    
    return metrics


def compute_auc(errors, thresholds=[5, 10, 20]):
    """Compute AUC metrics for given errors and thresholds."""
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    auc_metrics = {}
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        auc = np.trapz(y, x) / thr
        auc_metrics[f'auc@{thr}'] = float(auc * 100)

    return auc_metrics


def plot_metrics(results, output_dir):
    """Generate plots for AUC and mAcc metrics across checkpoints."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract steps and organize metrics
    steps = sorted(int(step) for step in results.keys())
    if not steps:
        print("No valid results to plot")
        return
        
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
    
    # Create AUC plot
    plt.figure(figsize=(10, 6))
    plt.title('AUC Metrics Across Training Steps')
    plt.plot(steps, metrics_data['auc@5'], 'b-', label='AUC@5')
    plt.plot(steps, metrics_data['auc@10'], 'g-', label='AUC@10')
    plt.plot(steps, metrics_data['auc@20'], 'r-', label='AUC@20')
    plt.xlabel('Training Step')
    plt.ylabel('AUC (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_metrics.png'), dpi=300)
    plt.close()
    
    # Create mAcc plot
    plt.figure(figsize=(10, 6))
    plt.title('mAcc Metrics Across Training Steps')
    plt.plot(steps, metrics_data['mAcc@5'], 'b-', label='mAcc@5')
    plt.plot(steps, metrics_data['mAcc@10'], 'g-', label='mAcc@10')
    plt.plot(steps, metrics_data['mAcc@20'], 'r-', label='mAcc@20')
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
    plt.plot(steps, metrics_data['auc@5'], 'b-', label='AUC@5')
    plt.plot(steps, metrics_data['auc@10'], 'g-', label='AUC@10')
    plt.plot(steps, metrics_data['auc@20'], 'r-', label='AUC@20')
    
    # Plot mAcc metrics with dashed lines
    plt.plot(steps, metrics_data['mAcc@5'], 'b--', label='mAcc@5')
    plt.plot(steps, metrics_data['mAcc@10'], 'g--', label='mAcc@10')
    plt.plot(steps, metrics_data['mAcc@20'], 'r--', label='mAcc@20')
    
    plt.xlabel('Training Step')
    plt.ylabel('Metric Value (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics.png'), dpi=300)
    plt.close()
    
    print(f"Plots saved to directory: {output_dir}")


def load_results_if_exists(output_file):
    """Load existing results from JSON file if it exists."""
    if os.path.exists(output_file) and os.path.isfile(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                return data.get("results", {})
        except Exception as e:
            print(f"Error loading existing results: {e}")
    return {}


def main():
    args = parse_args()
    
    # Ensure checkpoint directory exists
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Ensure output file is a file, not a directory
    output_path = args.output_file
    if os.path.exists(output_path) and os.path.isdir(output_path):
        output_path = os.path.join(output_path, "validation_results.json")
        print(f"Output path is a directory. Changed to {output_path}")
    
    # Create plots directory
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Load dataset
    try:
        dataset = CustomDataset(json_file=args.json_file, root_dir=args.dataset_dir)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Check if results already exist
    existing_results = load_results_if_exists(output_path)
    
    # Find all checkpoint files
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
        key=extract_step_from_filename
    )
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Initialize results dictionary with existing results
    results = existing_results
    
    # Process each checkpoint that hasn't been processed yet
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        step = extract_step_from_filename(checkpoint_file)
        
        # Skip if this checkpoint has already been processed
        if str(step) in results:
            print(f"Skipping already processed checkpoint: {checkpoint_file}")
            continue
        
        # Initialize XFeat model
        if args.use_star:
            xfeat = XFeat(top_k=args.top_k)
            matcher_fn = xfeat.match_xfeat_star
            model_type = "XFeat*"
        else:
            xfeat = XFeat()
            matcher_fn = xfeat.match_xfeat
            model_type = "XFeat"
        
        # Load checkpoint
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            
            # Load the checkpoint with CPU map_location to avoid CUDA errors
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
            # Fix the state dict keys (remove 'net.' prefix from expected keys)
            fixed_state_dict = {}
            for k, v in checkpoint.items():
                # The saved checkpoints don't have 'net.' prefix but the model expects it
                if not k.startswith('net.'):
                    fixed_state_dict[f'net.{k}'] = v
                else:
                    fixed_state_dict[k] = v
            
            # Load the fixed state dict
            xfeat.load_state_dict(fixed_state_dict, strict=False)
            print("Checkpoint loaded successfully with key remapping")
            
            # Run validation
            metrics = run_validation_for_checkpoint(
                checkpoint_path, matcher_fn, loader, args.ransac_thr
            )
            
            # Store results
            results[step] = {
                "file": checkpoint_file,
                "step": step,
                "model_type": model_type,
                "metrics": metrics
            }
            
            # Print current results
            print(f"\nResults for {checkpoint_file} (Step {step}):")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.2f}")
            print()
            
            # Save incremental results to avoid losing progress
            try:
                with open(output_path, 'w') as f:
                    json.dump({
                        "ransac_threshold": args.ransac_thr,
                        "model_type": model_type,
                        "results": results
                    }, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save incremental results: {e}")
            
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_file}: {e}")
    
    # Save final results to JSON
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "ransac_threshold": args.ransac_thr,
                "model_type": model_type,
                "results": results
            }, f, indent=2)
        
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        # Try to save to the current directory as a fallback
        fallback_path = "validation_results.json"
        try:
            with open(fallback_path, 'w') as f:
                json.dump({
                    "ransac_threshold": args.ransac_thr,
                    "model_type": model_type,
                    "results": results
                }, f, indent=2)
            print(f"Results saved to fallback location: {fallback_path}")
        except Exception as e2:
            print(f"Error saving to fallback location: {e2}")
    
    # Generate plots if we have results
    if results:
        plot_metrics(results, args.plots_dir)
        
        # Display best checkpoint
        best_step = max(results.keys(), key=lambda s: results[s]["metrics"]["auc@10"])
        best_checkpoint = results[best_step]
        print("\nBest checkpoint:")
        print(f"Step: {best_step}")
        print(f"File: {best_checkpoint['file']}")
        print("Metrics:")
        for metric_name, metric_value in best_checkpoint["metrics"].items():
            print(f"{metric_name}: {metric_value:.2f}")
    else:
        print("No results to plot.")


if __name__ == "__main__":
    main()
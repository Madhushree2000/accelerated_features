"""
Checkpoint validation script for XFeat model.
This script loads and evaluates all XFeat checkpoints in a specified directory.
"""

import os
import re
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import tqdm
import json
from pathlib import Path

# Import the necessary modules from your code
from modules.xfeat import XFeat
from paste import MegaDepth1500, compute_pose_error, compute_maa, tensor2bgr, run_pose_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Validate XFeat model checkpoints")
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help="Path to MegaDepth dataset root")
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


def main():
    args = parse_args()
    
    # Ensure checkpoint directory exists
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Load dataset
    dataset = MegaDepth1500(
        json_file='./assets/megadepth_1500.json',
        root_dir=f"{args.dataset_dir}/megadepth_test_1500"
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Find all checkpoint files
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
        key=extract_step_from_filename
    )
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Initialize results dictionary
    results = {}
    
    # Process each checkpoint
    for checkpoint_file in checkpoint_files:
        checkpoint_path = checkpoint_dir / checkpoint_file
        step = extract_step_from_filename(checkpoint_file)
        
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
            # Adjust this according to how your model loads checkpoints
            xfeat.load_state_dict(torch.load(checkpoint_path))
            
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
            
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_file}: {e}")
    
    # Save results to JSON
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump({
            "ransac_threshold": args.ransac_thr,
            "model_type": model_type,
            "results": results
        }, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Display best checkpoint
    if results:
        best_step = max(results.keys(), key=lambda s: results[s]["metrics"]["auc@10"])
        best_checkpoint = results[best_step]
        print("\nBest checkpoint:")
        print(f"Step: {best_step}")
        print(f"File: {best_checkpoint['file']}")
        print("Metrics:")
        for metric_name, metric_value in best_checkpoint["metrics"].items():
            print(f"{metric_name}: {metric_value:.2f}")


if __name__ == "__main__":
    main()
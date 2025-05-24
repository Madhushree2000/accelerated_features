import os
import re
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import json
import time
from pathlib import Path

# Import the necessary modules from your code
from modules.xfeat import XFeat
from modules.eval.batcheddatasettraj_04 import CustomDataset, compute_pose_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Validate XFeat model checkpoints with custom dataset")
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help="Path to dataset root containing the 36 folders")
    parser.add_argument('--json-file', type=str, required=True,
                        help="Path to JSON file with camera calibration and pose information")
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
    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of worker processes for data loading")
    return parser.parse_args()


def extract_step_from_filename(filename):
    """Extract step number from checkpoint filename."""
    match = re.search(r'_(\d+)\.pth$', filename)
    return int(match.group(1)) if match else 0


def run_validation_for_checkpoint(checkpoint_path, matcher_fn, loader, ransac_thr=2.5):
    """Run validation for a single checkpoint and return metrics."""
    pairs = []
    failed_count = 0
    
    for d in tqdm.tqdm(loader, desc="Processing pairs"):
        try:
            d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in d.items()}
            
            img0 = (d['image0'][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img1 = (d['image1'][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            src_pts, dst_pts = matcher_fn(img0, img1)
            
            del d['image0'], d['image1']
            torch.cuda.empty_cache()
            
            d = {k: v.cpu() if torch.is_tensor(v) else v for k, v in d.items()}
            
            src_pts = src_pts * d['scale0'].numpy()
            dst_pts = dst_pts * d['scale1'].numpy()
            
            if len(src_pts) < 8:
                failed_count += 1
                continue
                
            d.update({"pts0": src_pts, "pts1": dst_pts, 'ransac_thr': ransac_thr})
            compute_pose_error(d)
            
            # Critical fix: Check if pose error was computed
            if 't_err' not in d or 'R_err' not in d:
                failed_count += 1
                continue
                
            pairs.append(d)
        except Exception as e:
            failed_count += 1
            continue

    if not pairs:
        print(f"Warning: No valid pairs found for checkpoint {checkpoint_path}")
        return {
            'auc@5': 0.0,
            'auc@10': 0.0,
            'auc@20': 0.0,
            'mAcc@5': 0.0,
            'mAcc@10': 0.0,
            'mAcc@20': 0.0
        }

    # Calculate AUC and accuracy
    errors = []
    thresholds = [5, 10, 20]
    for p in pairs:
        et = p['t_err']
        er = p['R_err']
        errors.append(max(et, er))
    
    errors = np.array(errors)
    auc_metrics = compute_auc(errors, thresholds)

    # Calculate accuracy metrics
    acc_metrics = {}
    for t in thresholds:
        acc = (errors <= t).mean() * 100  # Convert to percentage
        acc_metrics[f'mAcc@{t}'] = float(acc)
    
    # Combine metrics
    metrics = {**auc_metrics, **acc_metrics}
    
    print(f"\nValidation results:")
    print(f"Processed {len(pairs)} pairs, failed on {failed_count} pairs")
    print(f"AUC@5: {metrics['auc@5']:.2f}%")
    print(f"AUC@10: {metrics['auc@10']:.2f}%")
    print(f"AUC@20: {metrics['auc@20']:.2f}%")
    print(f"mAcc@5: {metrics['mAcc@5']:.2f}%")
    print(f"mAcc@10: {metrics['mAcc@10']:.2f}%")
    print(f"mAcc@20: {metrics['mAcc@20']:.2f}%")
    
    return metrics


def compute_auc(errors, thresholds=[5, 10, 20]):
    """Compute AUC metrics for given errors and thresholds."""
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    auc_metrics = {}
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]] if last_index > 0 else [0]
        x = errors[:last_index] + [thr] if last_index > 0 else [thr]
        auc = np.trapz(y, x) / thr
        auc_metrics[f'auc@{thr}'] = float(auc * 100)  # Convert to percentage

    return auc_metrics


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    # Initialize dataset and data loader
    dataset = CustomDataset(json_file=args.json_file, root_dir=args.dataset_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Find and sort checkpoint files
    checkpoint_files = sorted(
        [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pth')],
        key=extract_step_from_filename
    )

    results = {}
    if not checkpoint_files:
        print(f"No checkpoint files found in {args.checkpoint_dir}")
        return
        
    for checkpoint_file in tqdm.tqdm(checkpoint_files, desc="Processing checkpoints"):
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
        step = extract_step_from_filename(checkpoint_file)
        
        if str(step) in results:
            continue

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = XFeat(top_k=args.top_k) if args.use_star else XFeat()
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(device).eval()
        
        # Create matcher function
        matcher_fn = model.match_xfeat_star if args.use_star else model.match_xfeat
        
        # Run validation
        metrics = run_validation_for_checkpoint(checkpoint_path, matcher_fn, loader, args.ransac_thr)
        
        # Store results
        results[str(step)] = {
            "file": checkpoint_file,
            "step": step,
            "metrics": metrics
        }

        # Save incremental results
        with open(args.output_file, 'w') as f:
            json.dump({
                "config": vars(args),
                "results": results
            }, f, indent=2)

    print(f"\nFinal results saved to {args.output_file}")


if __name__ == "__main__":
    main()
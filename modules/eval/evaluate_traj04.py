"""
Checkpoint evaluation and visualization script for XFeat model with Custom Dataset.
This script loads and evaluates XFeat checkpoints and generates plots of performance metrics
using the custom dataset with 36 folders of 12 images each.
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Validate and visualize XFeat model checkpoints with custom dataset")
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
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help="Directory to save visualization plots")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of worker processes for data loading")
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
    failed_count = 0
    
    for d in tqdm.tqdm(loader):
        try:
            # Move batch to GPU
            d = {k: v.to(device) if torch.is_tensor(v) else v for k,v in d.items()}
            
            # Convert images to numpy while still on GPU
            img0 = (d['image0'][0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            img1 = (d['image1'][0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            
            src_pts, dst_pts = matcher_fn(img0, img1)
            
            # Clean up GPU memory
            del d['image0'], d['image1']
            torch.cuda.empty_cache()
            
            # Move other tensors to CPU for processing
            d = {k: v.cpu() if torch.is_tensor(v) else v for k,v in d.items()}
            
            # Rescale keypoints
            src_pts = src_pts * d['scale0'].numpy()
            dst_pts = dst_pts * d['scale1'].numpy()
            
            if len(src_pts) < 8:
                print(f"Warning: Not enough matches ({len(src_pts)}), skipping pair")
                failed_count += 1
                continue
                
            d.update({"pts0": src_pts, "pts1": dst_pts, 'ransac_thr': ransac_thr})
            compute_pose_error(d)
            pairs.append(d)
        except Exception as e:
            print(f"Error processing image pair: {e}")
            failed_count += 1
            continue


def compute_auc(errors, thresholds=[5, 10, 20]):
    """Compute AUC metrics for given errors and thresholds."""
    if len(errors) == 0:
        return {f'auc@{thr}': 0.0 for thr in thresholds}
        
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
            if metric in step_metrics:
                metrics_data[metric].append(step_metrics[metric])
            else:
                metrics_data[metric].append(0)  # Default if metric is missing
    
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
    
    # Create per-scene performance plot for last checkpoint (if available)
    latest_result = results[str(steps[-1])]
    if 'scene_metrics' in latest_result['metrics']:
        scene_metrics = latest_result['metrics']['scene_metrics']
        scene_ids = sorted(scene_metrics.keys())
        
        if scene_ids:
            # Extract mAcc@10 for each scene
            scene_accs = [scene_metrics[scene]['mAcc@10'] for scene in scene_ids]
            scene_counts = [scene_metrics[scene]['count'] for scene in scene_ids]
            
            # Generate scene labels with count
            scene_labels = [f"{scene.replace('scene_', '')} ({count})" for scene, count in zip(scene_ids, scene_counts)]
            
            # Sort by accuracy
            sorted_indices = np.argsort(scene_accs)
            sorted_accs = [scene_accs[i] for i in sorted_indices]
            sorted_labels = [scene_labels[i] for i in sorted_indices]
            
            # Plot the per-scene performance
            plt.figure(figsize=(14, 8))
            plt.title(f'Per-Scene Performance (mAcc@10) - Checkpoint {steps[-1]}')
            plt.barh(sorted_labels, sorted_accs)
            plt.axvline(x=np.mean(scene_accs), color='r', linestyle='--', label=f'Average: {np.mean(scene_accs):.1f}%')
            plt.xlabel('mAcc@10 (%)')
            plt.ylabel('Scene ID (count)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'scene_performance.png'), dpi=300)
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


def validate_json_structure(json_file):
    """
    Validates that the JSON file has the expected structure compatible with the dataset.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if len(data) == 0:
            print("Warning: JSON file contains no entries")
            return False
            
        required_keys = ['pair_names', 'size0_hw', 'size1_hw', 'K0', 'K1', 'T_0to1', 'scene_id']
        sample = data[0]
        
        for key in required_keys:
            if key not in sample:
                print(f"Error: Required key '{key}' missing from JSON data")
                return False
                
        # Check pair_names structure
        if not (isinstance(sample['pair_names'], list) and len(sample['pair_names']) == 2):
            print("Error: 'pair_names' should be a list with 2 elements")
            return False
            
        print(f"JSON validation successful: {len(data)} entries found")
        return True
        
    except Exception as e:
        print(f"Error validating JSON file: {e}")
        return False


def main():
    # Parse arguments and initialize CUDA
    args = parse_args()
    
    # Verify CUDA availability and print device info
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU acceleration.")
    device = torch.device('cuda')
    print(f"\nUsing device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: Allocated={torch.cuda.memory_allocated()/1024**2:.2f}MB, "
          f"Reserved={torch.cuda.memory_reserved()/1024**2:.2f}MB\n")

    # Validate JSON structure
    if not validate_json_structure(args.json_file):
        print("JSON validation failed. Please check your JSON file format.")
        return

    # Setup directories and paths
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    output_path = args.output_file
    if os.path.exists(output_path) and os.path.isdir(output_path):
        output_path = os.path.join(output_path, "validation_results.json")
        print(f"Output path is a directory. Changed to {output_path}")

    os.makedirs(args.plots_dir, exist_ok=True)

    # Initialize dataset and data loader with optimized settings
    try:
        dataset = CustomDataset(json_file=args.json_file, root_dir=args.dataset_dir)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2 if args.batch_size > 1 else None
        )
        print(f"Dataset loaded with {len(dataset)} image pairs")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load existing results if available
    existing_results = load_results_if_exists(output_path)
    
    # Find and sort checkpoint files
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
        key=extract_step_from_filename
    )
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    results = existing_results

    # Process each checkpoint
    for checkpoint_file in tqdm.tqdm(checkpoint_files, desc="Processing checkpoints"):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        step = extract_step_from_filename(checkpoint_file)
        
        if str(step) in results:
            print(f"Skipping already processed checkpoint: {checkpoint_file}")
            continue
        
        # Initialize model with proper CUDA handling
        try:
            # Load checkpoint to CPU first
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Initialize model based on arguments
            if args.use_star:
                model = XFeat(top_k=args.top_k)
                model_type = "XFeat*"
            else:
                model = XFeat()
                model_type = "XFeat"
            
            # Handle state dict keys and load model
            state_dict = {k.replace('module.', '').replace('net.', ''): v 
                         for k,v in checkpoint.items()}
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device).eval()
            
            # Create matcher function with proper CUDA handling
            def matcher_fn(img0, img1):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    if args.use_star:
                        return model.match_xfeat_star(img0, img1)
                    return model.match_xfeat(img0, img1)

            # Run validation with memory monitoring
            torch.cuda.empty_cache()
            metrics = run_validation_for_checkpoint(
                checkpoint_path, matcher_fn, loader, args.ransac_thr
            )
            
            # Store results
            results[str(step)] = {
                "file": checkpoint_file,
                "step": step,
                "model_type": model_type,
                "metrics": metrics
            }

            # Print and save incremental results
            print(f"\nResults for {checkpoint_file} (Step {step}):")
            print(f"GPU Memory after processing: Allocated={torch.cuda.memory_allocated()/1024**2:.2f}MB")
            for metric, value in metrics.items():
                if metric != 'scene_metrics':
                    print(f"{metric}: {value:.2f}")

            # Save incremental results
            try:
                with open(output_path, 'w') as f:
                    json.dump({
                        "config": {
                            "ransac_threshold": args.ransac_thr,
                            "model_type": model_type,
                            "device": str(device),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        },
                        "results": results
                    }, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save incremental results: {e}")

        except Exception as e:
            print(f"\nError processing checkpoint {checkpoint_file}: {e}")
            print(f"GPU Memory at error: Allocated={torch.cuda.memory_allocated()/1024**2:.2f}MB")
            torch.cuda.empty_cache()

    # Final results processing
    try:
        # Save final results with additional metadata
        final_results = {
            "config": {
                "dataset": args.dataset_dir,
                "matcher": model_type,
                "ransac_threshold": args.ransac_thr,
                "device": str(device),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "statistics": {
                "total_checkpoints": len(checkpoint_files),
                "processed_checkpoints": len(results),
                "success_rate": len(results)/len(checkpoint_files)*100
            },
            "results": results
        }

        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\nFinal results saved to {output_path}")

        # Generate plots and show best checkpoint
        if results:
            plot_metrics(results, args.plots_dir)
            best_step = max(results.keys(), key=lambda s: results[s]["metrics"]["auc@10"])
            best = results[best_step]
            print("\n=== Best Checkpoint ===")
            print(f"Step: {best_step} | File: {best['file']}")
            print("Metrics:")
            for metric, value in best["metrics"].items():
                if metric != 'scene_metrics':
                    print(f"{metric}: {value:.2f}")
        else:
            print("No valid results to plot.")

    except Exception as e:
        print(f"\nError in final processing: {e}")
        # Emergency save if final save fails
        try:
            with open("emergency_results.json", 'w') as f:
                json.dump({"results": results}, f)
            print("Emergency results saved to emergency_results.json")
        except:
            print("Could not save emergency results")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    # Set up CUDA optimization flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    main()
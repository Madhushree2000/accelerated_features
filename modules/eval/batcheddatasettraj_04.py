"""
	Adapted from "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Camera pose metrics adapted from LoFTR https://github.com/zju3dv/LoFTR/blob/master/src/utils/metrics.py
    
    Modified to work with a custom dataset structure of 36 folders with 12 images each.
"""

import argparse, glob, sys, os, time
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import poselib
import json
import copy

import tqdm

# Disable scientific notation
np.set_printoptions(suppress=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark
else:
    device = torch.device('cpu')
    print("Warning: CUDA not available, falling back to CPU")

class CustomDataset(Dataset):
    """
    Custom dataset loader for a dataset with 36 folders, each containing 12 images.
    The camera poses & metadata are stored in a formatted json similar to MegaDepth1500.
    """
    def __init__(self, json_file, root_dir):
        # Load the info & calibration from the JSON
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.root_dir = root_dir

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
            f"Dataset {self.root_dir} does not exist! Please check the path provided.")

        # Print dataset statistics
        num_folders = len(set([item['scene_id'] for item in self.data])) if len(self.data) > 0 else 0
        print(f"Dataset loaded with {len(self.data)} image pairs across {num_folders} folders")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.data[idx])
        
        h1, w1 = data['size0_hw']
        h2, w2 = data['size1_hw']
        
        scene_id = data['scene_id']
        image0_path = os.path.join(self.root_dir, scene_id, data['pair_names'][0])
        image1_path = os.path.join(self.root_dir, scene_id, data['pair_names'][1])
        
        # Read and resize images
        image0 = cv2.resize(cv2.imread(image0_path), (w1, h1))
        image1 = cv2.resize(cv2.imread(image1_path), (w2, h2))
        
        # Keep as CPU tensors here, will move to GPU later
        data['image0'] = torch.tensor(image0.astype(np.float32)/255).permute(2,0,1)
        data['image1'] = torch.tensor(image1.astype(np.float32)/255).permute(2,0,1)
        
        for k,v in data.items():
            if k not in ('dataset_name', 'scene_id', 'pair_id', 'pair_names', 'size0_hw', 'size1_hw', 'image0', 'image1'):
                data[k] = torch.tensor(np.array(v, dtype=np.float32))
        
        return data


################################# Metrics #####################################

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def intrinsics_to_camera(K, distortion=None, model="OPENCV_FISHEYE"):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    if distortion is None:
        distortion = [0, 0, 0, 0]  # Default if not provided

    return {
        "model": model,
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py] + distortion,
    }


def estimate_pose_poselib(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    distortion = [0.04816514, 0.17073599, -0.30080163, 0.47220233]
    M, info = poselib.estimate_relative_pose(
        kpts0, kpts1,
        intrinsics_to_camera(K0, distortion=distortion, model="OPENCV_FISHEYE"),
        intrinsics_to_camera(K1, distortion=distortion, model="OPENCV_FISHEYE"),
        {"max_epipolar_error": thresh,
         "success_prob": conf,
         "min_iterations": 20,
         "max_iterations": 1_000},
    )

    R, t, inl = M.R, M.t, info["inliers"]
    inl = np.array(inl)
    ret = (R, t, inl)

    return ret, (kpts0, kpts1)


def tensor2bgr(t):
    return (t.cpu()[0].permute(1,2,0).numpy()*255).astype(np.uint8)


def compute_pose_error(pair):
    pixel_thr = 1.0 if 'ransac_thr' not in pair else pair['ransac_thr']
    conf = 0.99999
    pair.update({'R_err':  np.inf, 't_err': np.inf, 'inliers': []})

    pts0 = pair['pts0']
    pts1 = pair['pts1']
    K0 = pair['K0'].numpy()[0]  # Already on CPU from previous step
    K1 = pair['K1'].numpy()[0]
    T_0to1 = pair['T_0to1'].numpy()[0]

    ret, corrs = estimate_pose_poselib(pts0, pts1, K0, K1, pixel_thr, conf=conf)

    if ret is not None:
        R, t, inliers = ret
        print(f"Estimated pose for pair {pair['pair_id']}: R = {R}, t = {t}, inliers = {len(inliers)}")
        print(f"Ground truth pose for pair {pair['pair_id']}: T_0to1 = {T_0to1}")
        t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
        pair['R_err'] = R_err
        pair['t_err'] = t_err
        print(f"Pair {pair['pair_id']}: R_err = {R_err:.2f}, t_err = {t_err:.2f}, inliers = {len(inliers)}")

def error_auc(errors, thresholds=[5, 10, 20]):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []

    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def compute_maa(pairs, thresholds=[5, 10, 20]):
    print("auc / mAcc on %d pairs" % (len(pairs)))
    errors = []

    for p in pairs:
        et = p['t_err']
        er = p['R_err']
        errors.append(max(et, er))

    d_err_auc = error_auc(errors)

    for k,v in d_err_auc.items():
        print(k, ': ', '%.1f'%(v*100))

    errors = np.array(errors)

    for t in thresholds:
        acc = (errors <= t).sum() / len(errors)
        print("mAcc@%d: %.1f "%(t, acc*100))
    
#     Additionally, compute per-folder performance
#     if len(pairs) > 0 and 'scene_id' in pairs[0]:
#         print("\nPer-folder performance:")
#         folders = {}
#         for p in pairs:
#             scene_id = p['scene_id']
#             if scene_id not in folders:
#                 folders[scene_id] = []
#             folders[scene_id].append(max(p['t_err'], p['R_err']))
        
#         for folder, errors in folders.items():
#             errors = np.array(errors)
#             acc_10 = (errors <= 10).sum() / len(errors)
#             print(f"Folder {folder}: mAcc@10: {acc_10*100:.1f}% ({len(errors)} pairs)")


# @torch.inference_mode()
# def run_pose_benchmark(matcher_fn, loader, ransac_thr=2.5):
#     """
#         Run relative pose estimation benchmark using a specified matcher function and data loader.

#         Parameters
#         ----------
#         matcher_fn : callable
#             The matching function to be evaluated for pose estimation. It should accept two np.array RGB images (H,W,3)
#             and return mkpts_0, mkpts_1 which are np.array(N,2) matching coordinates.
        
#         loader : iterable
#             Data loader that provides batches of data. Each batch should contain two images, along 
#             with their groundtruth camera poses.
        
#         ransac_thr : float, optional, default=2.5
#             The RANSAC threshold for considering a point as an inlier in pixels.
#     """
#     pairs = []
#     cnt = 0
#     failed_pairs = 0
    
#     for d in tqdm.tqdm(loader):
#         try:
#             # Move batch to GPU
#             d = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in d.items()}

#             # Convert images to numpy while still on GPU
#             img0 = d['image0'][0].permute(1,2,0).cpu().numpy()*255
#             img1 = d['image1'][0].permute(1,2,0).cpu().numpy()*255

#             src_pts, dst_pts = matcher_fn(img0.astype(np.uint8), img1.astype(np.uint8))
            
#             # Clean up GPU memory
#             del d['image0'], d['image1']
#             torch.cuda.empty_cache()
            
#             # Move other tensors to CPU for processing
#             d = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k,v in d.items()}
            
#             # Rescale keypoints
#             src_pts = src_pts * d['scale0'].numpy()
#             dst_pts = dst_pts * d['scale1'].numpy()
            
#             if len(src_pts) < 8:
#                 print(f"Warning: Not enough matches ({len(src_pts)}) for pair {cnt}, skipping...")
#                 failed_pairs += 1
#                 continue
                
#             d.update({"pts0": src_pts, "pts1": dst_pts, 'ransac_thr': ransac_thr})
#             compute_pose_error(d)
#             pairs.append(d)
            
#         except Exception as e:
#             print(f"Error processing pair {cnt}: {e}")
#             failed_pairs += 1
            
#         cnt += 1

@torch.inference_mode()
def run_pose_benchmark(matcher_fn, loader, ransac_thr=2.5):
    """
        Run relative pose estimation benchmark using a specified matcher function and data loader.

        Parameters
        ----------
        matcher_fn : callable
            The matching function to be evaluated for pose estimation. It should accept two np.array RGB images (H,W,3)
            and return mkpts_0, mkpts_1 which are np.array(N,2) matching coordinates.
        
        loader : iterable
            Data loader that provides batches of data. Each batch should contain two images, along 
            with their groundtruth camera poses.
        
        ransac_thr : float, optional, default=2.5
            The RANSAC threshold for considering a point as an inlier in pixels.
    """


    pairs = []
    cnt = 0
    for d in tqdm.tqdm(loader):
        d_error = {}
        src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']))

        #delete images to avoid OOM, happens in low mem machines
        del d['image0']
        del d['image1']

        #rescale kpts
        src_pts = src_pts * d['scale0'].numpy()
        dst_pts = dst_pts * d['scale1'].numpy()
        d.update({"pts0":src_pts, "pts1": dst_pts,'ransac_thr': ransac_thr})
        compute_pose_error(d)
        pairs.append(d)
        cnt+=1

    compute_maa(pairs)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run pose benchmark with matcher on custom dataset")
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help="Path to dataset root containing the 36 folders")
    parser.add_argument('--json-file', type=str, required=True,
                        help="Path to JSON file with camera calibration and pose information")
    parser.add_argument('--matcher', type=str, choices=['xfeat', 'xfeat-star', 'alike'], default='xfeat',
                        help="Matcher to use (xfeat, xfeat-star, or alike)")
    parser.add_argument('--ransac-thr', type=float, default=2.5,
                        help="RANSAC threshold value in pixels (default: 2.5)")
    parser.add_argument('--batch-size', type=int, default=1,
                        help="Batch size for data loader (default: 1)")
    return parser.parse_args()


if __name__ == '__main__':
    # Initialize CUDA and multiprocessing
    import torch.multiprocessing as mp
    torch.backends.cudnn.benchmark = True
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    
    # Verify CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU acceleration.")
    device = torch.device('cuda')
    print(f"\nRunning on {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory - Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB, "
          f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f}MB\n")

    # Validate JSON file structure
    if not validate_json_structure(args.json_file):
        print("JSON validation failed. Please check the format of your JSON file.")
        sys.exit(1)
    
    # Initialize dataset with proper device handling
    try:
        dataset = CustomDataset(json_file=args.json_file, root_dir=args.dataset_dir)
        
        # Optimized DataLoader configuration
        loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=min(4, os.cpu_count()),  # Dynamic worker count
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2 if args.batch_size > 1 else None
        )
        print(f"Dataset loaded with {len(dataset)} image pairs")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        sys.exit(1)

    # Benchmark with selected matcher - CUDA optimized
    results = []
    try:
        if args.matcher == 'xfeat':
            print("Running benchmark for XFeat with CUDA optimization..")
            from modules.xfeat import XFeat
            xfeat = XFeat().to(device).eval()  # Explicit device and eval mode
            
            def wrapped_matcher(img0, img1):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    return xfeat.match_xfeat(img0, img1)
                    
            results = run_pose_benchmark(
                matcher_fn=wrapped_matcher, 
                loader=loader, 
                ransac_thr=args.ransac_thr
            )
            
        elif args.matcher == 'xfeat-star':
            from modules.xfeat import XFeat
            print("Running benchmark for XFeat* with CUDA optimization..")
            xfeat = XFeat(top_k=args.top_k).to(device).eval()
            
            def wrapped_matcher(img0, img1):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    return xfeat.match_xfeat_star(img0, img1)
                    
            results = run_pose_benchmark(
                matcher_fn=wrapped_matcher,
                loader=loader,
                ransac_thr=args.ransac_thr
            )
            
        elif args.matcher == 'alike':
            from third_party import alike_wrapper as alike
            print("Running benchmark for ALIKE with CUDA optimization..")
            alike_model = alike.ALike().to(device).eval()
            
            def wrapped_matcher(img0, img1):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    return alike_model.match_alike(img0, img1)
                    
            results = run_pose_benchmark(
                matcher_fn=wrapped_matcher,
                loader=loader,
                ransac_thr=args.ransac_thr
            )
            
    except Exception as e:
        print(f"Error during benchmark: {e}")
        print(f"GPU Memory at error: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        torch.cuda.empty_cache()
        sys.exit(1)

    # Save results with enhanced metadata
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = f"results_{args.matcher}_{timestamp}.json"
    
    serializable_results = {
        "metadata": {
            "matcher": args.matcher,
            "ransac_threshold": args.ransac_thr,
            "device": str(device),
            "timestamp": timestamp,
            "dataset": os.path.basename(args.dataset_dir),
            "num_pairs": len(results)
        },
        "pair_errors": [
            {
                'scene_id': p.get('scene_id', 'unknown'),
                'pair_id': p.get('pair_id', 'unknown'),
                'R_err': float(p['R_err']),
                't_err': float(p['t_err']),
                'num_matches': len(p.get('pts0', [])) if 'pts0' in p else 0
            } for p in results
        ],
        "statistics": {
            "mean_R_err": np.mean([p['R_err'] for p in results]),
            "mean_t_err": np.mean([p['t_err'] for p in results]),
            "success_rate": len(results)/len(dataset)*100
        }
    }

    try:
        with open(result_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults successfully saved to {result_file}")
        print(f"Final GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    except Exception as e:
        print(f"Error saving results: {e}")
        # Emergency save
        try:
            with open(f"emergency_results_{timestamp}.json", 'w') as f:
                json.dump({"pair_errors": serializable_results["pair_errors"]}, f)
            print("Emergency results saved")
        except:
            print("Failed to save emergency results")

    # Clean up
    torch.cuda.empty_cache()
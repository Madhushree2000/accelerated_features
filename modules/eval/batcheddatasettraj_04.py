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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        # Get the scene_id (batch folder)
        scene_id = data['scene_id']
        
        # Construct the full path to the images using scene_id as the batch folder
        image0_path = os.path.join(self.root_dir, scene_id, data['pair_names'][0])
        image1_path = os.path.join(self.root_dir, scene_id, data['pair_names'][1])
        
        if not os.path.exists(image0_path):
            raise FileNotFoundError(f"Image not found: {image0_path}")
        if not os.path.exists(image1_path):
            raise FileNotFoundError(f"Image not found: {image1_path}")

        # Read and resize images
        image0 = cv2.resize(cv2.imread(image0_path), (w1, h1))
        image1 = cv2.resize(cv2.imread(image1_path), (w2, h2))

        data['image0'] = torch.tensor(image0.astype(np.float32)/255).permute(2,0,1).to(device)
        data['image1'] = torch.tensor(image1.astype(np.float32)/255).permute(2,0,1).to(device)

        for k,v in data.items():
            if k not in ('dataset_name', 'scene_id', 'pair_id', 'pair_names', 'size0_hw', 'size1_hw', 'image0', 'image1'):
                data[k] = torch.tensor(np.array(v, dtype=np.float32)).to(device)

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


def intrinsics_to_camera(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }

def estimate_pose_poselib(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    M, info = poselib.estimate_relative_pose(
        kpts0, kpts1,
        intrinsics_to_camera(K0),
        intrinsics_to_camera(K1),
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
    """ 
    Input:
        pair (dict):{
            "pts0": ndrray(N,2)
            "pts1": ndrray(N,2)
            "K0": ndrray(3,3)
            "K1": ndrray(3,3)
            "T_0to1": ndrray(4,4)

        }
    Update:
        pair (dict):{
            "R_err" List[float]: [N]
            "t_err" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = 1.0 if 'ransac_thr' not in pair else pair['ransac_thr']
    conf = 0.99999
    pair.update({'R_err':  np.inf, 't_err': np.inf, 'inliers': []})

    pts0 = pair['pts0']
    pts1 = pair['pts1']
    K0 = pair['K0'].to(device).cpu().numpy()[0]
    K1 = pair['K1'].to(device).cpu().numpy()[0]
    T_0to1 = pair['T_0to1'].to(device).cpu().numpy()[0]

    ret, corrs = estimate_pose_poselib(pts0, pts1, K0, K1, pixel_thr, conf=conf)

    if ret is not None:
        R, t, inliers = ret

        t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)

        pair['R_err'] = R_err
        pair['t_err'] = t_err


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
    
    # Additionally, compute per-folder performance
    if len(pairs) > 0 and 'scene_id' in pairs[0]:
        print("\nPer-folder performance:")
        folders = {}
        for p in pairs:
            scene_id = p['scene_id']
            if scene_id not in folders:
                folders[scene_id] = []
            folders[scene_id].append(max(p['t_err'], p['R_err']))
        
        for folder, errors in folders.items():
            errors = np.array(errors)
            acc_10 = (errors <= 10).sum() / len(errors)
            print(f"Folder {folder}: mAcc@10: {acc_10*100:.1f}% ({len(errors)} pairs)")


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
    failed_pairs = 0
    
    for d in tqdm.tqdm(loader):
        try:
            src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']))
            
            # Delete images to avoid OOM
            del d['image0']
            del d['image1']
            
            # Rescale keypoints
            src_pts = src_pts * d['scale0'].numpy()
            dst_pts = dst_pts * d['scale1'].numpy()
            
            # Skip if no matches found
            if len(src_pts) < 8:
                print(f"Warning: Not enough matches ({len(src_pts)}) for pair {cnt}, skipping...")
                failed_pairs += 1
                continue
                
            d.update({"pts0": src_pts, "pts1": dst_pts, 'ransac_thr': ransac_thr})
            compute_pose_error(d)
            pairs.append(d)
            
        except Exception as e:
            print(f"Error processing pair {cnt}: {e}")
            failed_pairs += 1
            
        cnt += 1

    print(f"Processed {cnt} pairs, {failed_pairs} failed")
    compute_maa(pairs)
    
    return pairs  # Return pairs for possible further analysis


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

    args = parse_args()
    
    # Validate JSON file structure
    if not validate_json_structure(args.json_file):
        print("JSON validation failed. Please check the format of your JSON file.")
        sys.exit(1)
    
    # Initialize dataset
    dataset = CustomDataset(json_file=args.json_file, root_dir=args.dataset_dir)
    
    # Create data loader
    loader = DataLoader(dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False,
                        num_workers=4)
    
    # Run benchmark with selected matcher
    if args.matcher == 'xfeat':
        print("Running benchmark for XFeat..")
        from modules.xfeat import XFeat
        xfeat = XFeat()
        results = run_pose_benchmark(matcher_fn=xfeat.match_xfeat, 
                                     loader=loader, 
                                     ransac_thr=args.ransac_thr)
    
    elif args.matcher == 'xfeat-star':
        from modules.xfeat import XFeat
        print("Running benchmark for XFeat*..")
        xfeat = XFeat(top_k=10_000)
        results = run_pose_benchmark(matcher_fn=xfeat.match_xfeat_star, 
                                     loader=loader, 
                                     ransac_thr=args.ransac_thr)
    
    elif args.matcher == 'alike':
        from third_party import alike_wrapper as alike
        print("Running benchmark for ALIKE..")
        results = run_pose_benchmark(matcher_fn=alike.match_alike, 
                                     loader=loader, 
                                     ransac_thr=args.ransac_thr)
    
    # Save results to file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = f"results_{args.matcher}_{timestamp}.json"
    
    # Convert results to serializable format
    serializable_results = {}
    serializable_results['matcher'] = args.matcher
    serializable_results['ransac_thr'] = args.ransac_thr
    serializable_results['num_pairs'] = len(results)
    
    # Save per-pair error metrics
    pair_errors = []
    for p in results:
        pair_errors.append({
            'scene_id': p['scene_id'] if 'scene_id' in p else 'unknown',
            'pair_id': p['pair_id'] if 'pair_id' in p else 'unknown',
            'R_err': float(p['R_err']),
            't_err': float(p['t_err'])
        })
    serializable_results['pair_errors'] = pair_errors
    
    with open(result_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {result_file}")
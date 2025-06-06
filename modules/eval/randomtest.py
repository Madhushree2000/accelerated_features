def run_validation(dataset_dir, json_file, matcher_type="xfeat", ransac_thr=2.5, batch_size=1):
    """
    Run XFeat validation on custom dataset with specified parameters.
    
    Args:
        dataset_dir (str): Path to dataset root containing the 36 folders
        json_file (str): Path to JSON file with camera calibration and pose information
        matcher_type (str): Matcher to use ('xfeat', 'xfeat-star', or 'alike')
        ransac_thr (float): RANSAC threshold value in pixels
        batch_size (int): Batch size for data loader
    """
    # Initialize CUDA and multiprocessing
    import torch.multiprocessing as mp
    torch.backends.cudnn.benchmark = True
    mp.set_start_method('spawn', force=True)
    
    # Verify CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU acceleration.")
    device = torch.device('cuda')
    print(f"\nRunning on {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory - Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB, "
          f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f}MB\n")

    # Validate JSON file structure
    if not validate_json_structure(json_file):
        print("JSON validation failed. Please check the format of your JSON file.")
        sys.exit(1)
    
    # Initialize dataset with proper device handling
    try:
        dataset = CustomDataset(json_file=json_file, root_dir=dataset_dir)
        
        # Optimized DataLoader configuration
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2 if batch_size > 1 else None
        )
        print(f"Dataset loaded with {len(dataset)} image pairs")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        sys.exit(1)

    # Benchmark with selected matcher - CUDA optimized
    results = []
    try:
        if matcher_type == 'xfeat':
            print("Running benchmark for XFeat with CUDA optimization..")
            from modules.xfeat import XFeat
            xfeat = XFeat().to(device).eval()
            
            def wrapped_matcher(img0, img1):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    return xfeat.match_xfeat(img0, img1)
                    
            results = run_pose_benchmark(
                matcher_fn=wrapped_matcher, 
                loader=loader, 
                ransac_thr=ransac_thr
            )
            
        elif matcher_type == 'xfeat-star':
            from modules.xfeat import XFeat
            print("Running benchmark for XFeat* with CUDA optimization..")
            xfeat = XFeat().to(device).eval()
            
            def wrapped_matcher(img0, img1):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    return xfeat.match_xfeat_star(img0, img1)
                    
            results = run_pose_benchmark(
                matcher_fn=wrapped_matcher,
                loader=loader,
                ransac_thr=ransac_thr
            )
            
        elif matcher_type == 'alike':
            from third_party import alike_wrapper as alike
            print("Running benchmark for ALIKE with CUDA optimization..")
            alike_model = alike.ALike().to(device).eval()
            
            def wrapped_matcher(img0, img1):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    return alike_model.match_alike(img0, img1)
                    
            results = run_pose_benchmark(
                matcher_fn=wrapped_matcher,
                loader=loader,
                ransac_thr=ransac_thr
            )
            
    except Exception as e:
        print(f"Error during benchmark: {e}")
        print(f"GPU Memory at error: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        torch.cuda.empty_cache()
        sys.exit(1)

    # Save results with enhanced metadata
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = f"results_{matcher_type}_{timestamp}.json"
    
    serializable_results = {
        "metadata": {
            "matcher": matcher_type,
            "ransac_threshold": ransac_thr,
            "device": str(device),
            "timestamp": timestamp,
            "dataset": os.path.basename(dataset_dir),
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
        return serializable_results
    except Exception as e:
        print(f"Error saving results: {e}")
        # Emergency save
        try:
            with open(f"emergency_results_{timestamp}.json", 'w') as f:
                json.dump({"pair_errors": serializable_results["pair_errors"]}, f)
            print("Emergency results saved")
        except:
            print("Failed to save emergency results")
        return None
    finally:
        # Clean up
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # Example usage with parameters
    dataset_dir = "/path/to/your_dataset"      # Replace with your actual path
    json_file = "/path/to/your_metadata.json"  # Replace with your actual path
    matcher_type = "xfeat"                     # Choose: 'xfeat', 'xfeat-star', 'alike'
    ransac_thr = 2.5                          # RANSAC threshold in pixels
    batch_size = 1                            # Batch size for processing
    
    # Run the validation
    results = run_validation(
        dataset_dir=dataset_dir,
        json_file=json_file, 
        matcher_type=matcher_type,
        ransac_thr=ransac_thr,
        batch_size=batch_size
    )
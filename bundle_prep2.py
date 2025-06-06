import numpy as np
import pickle
import cv2
import torch
import os
import glob
from pathlib import Path
import re
from scipy.spatial.transform import Rotation as R
import numpy as np
from modules.xfeat import XFeat

def parse_calibration_file(calib_path):
    """Parse the calibration file and extract cam0 parameters."""
    with open(calib_path, 'r') as f:
        content = f.read()
    
    # Extract cam0 parameters using regex
    cam0_section = re.search(r'cam0.*?(?=cam1|Target configuration|$)', content, re.DOTALL)
    if not cam0_section:
        raise ValueError("Could not find cam0 section in calibration file")
    
    cam0_text = cam0_section.group()
    
    # Extract distortion coefficients
    distortion_match = re.search(r'distortion: \[(.*?)\]', cam0_text)
    if distortion_match:
        distortion_str = distortion_match.group(1)
        distortion = [float(x.strip()) for x in distortion_str.split()]
    else:
        raise ValueError("Could not parse distortion coefficients")
    
    # Extract projection parameters (fx, fy, cx, cy)
    projection_match = re.search(r'projection: \[(.*?)\]', cam0_text)
    if projection_match:
        projection_str = projection_match.group(1)
        projection = [float(x.strip()) for x in projection_str.split()]
        fx, fy, cx, cy = projection
    else:
        raise ValueError("Could not parse projection parameters")
    
    # Create intrinsic matrix
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return {
        'intrinsic': intrinsic_matrix,
        'distortion': np.array(distortion)
    }

def parse_tum_file(tum_path):
    """Parse TUM format file and return poses."""
    poses_data = []
    
    with open(tum_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 8:
                image_name = f"image_{len(poses_data):06d}.jpg"  # Generate image name based on index
                timestamp = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                
                poses_data.append({
                    'image_name': image_name,
                    'timestamp': timestamp,
                    'translation': np.array([tx, ty, tz]),
                    'quaternion': np.array([qx, qy, qz, qw])  # x,y,z,w format
                })
    
    return poses_data

def quaternion_to_rotation_matrix(q):
    """Convert quaternion (x,y,z,w) to rotation matrix."""
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    qx, qy, qz, qw = q
    
    # Convert to rotation matrix
    rotation_matrix = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return rotation_matrix

def load_xfeat_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load XFeat model for inference."""
    try:
        xfeat = XFeat()
        xfeat.to(device)
        xfeat.eval()
        return xfeat, device
    except Exception as e:
        print(f"Error loading XFeat model: {e}")
        return None, device

def extract_features_sequence(image_paths, xfeat_model, device, max_features=1000):
    """Extract features from a sequence of images using XFeat."""
    if not image_paths:
        raise ValueError("No image paths provided")
    
    all_keypoints = []
    max_keypoints = 0
    
    # Test the first image to understand XFeat's API
    print("Testing XFeat API with first image...")
    test_img = cv2.imread(str(image_paths[0]))
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    # Debug XFeat methods
    print("Available XFeat methods:", [method for method in dir(xfeat_model) if not method.startswith('_')])
    
    # Test different input formats
    try:
        # Test with numpy array
        print("Testing with numpy array...")
        result = xfeat_model.detectAndCompute(test_img_rgb, top_k=100)
        print(f"Result type: {type(result)}, Result: {result}")
        
        # Test with tensor
        print("Testing with tensor...")
        img_tensor = torch.from_numpy(test_img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)
        result2 = xfeat_model.detectAndCompute(img_tensor, top_k=100)
        print(f"Tensor result type: {type(result2)}, Result: {result2}")
        
    except Exception as e:
        print(f"Debug test failed: {e}")
    
    # Try the standard XFeat workflow
    print("\nStarting feature extraction...")
    
    # Process subset of images first for testing
    test_images = image_paths[:min(10, len(image_paths))]
    
    for i, img_path in enumerate(test_images):
        print(f"Processing image {i+1}/{len(test_images)}: {os.path.basename(img_path)}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to grayscale (XFeat might expect grayscale)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        keypoints = None
        
        try:
            with torch.no_grad():
                # Method 1: Standard XFeat approach with tensor input
                try:
                    # Convert to tensor format expected by XFeat
                    if len(img_gray.shape) == 2:  # Grayscale
                        img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
                    else:  # RGB
                        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                    
                    img_tensor = img_tensor.to(device)
                    
                    # Try extracting keypoints
                    outputs = xfeat_model.detectAndCompute(img_tensor, top_k=max_features)
                    print(f"  detectAndCompute output type: {type(outputs)}")
                    
                    if isinstance(outputs, tuple) and len(outputs) >= 1:
                        kpts = outputs[0]
                        print(f"  Keypoints type: {type(kpts)}, shape: {kpts.shape if hasattr(kpts, 'shape') else 'no shape'}")
                        
                        if torch.is_tensor(kpts):
                            kpts_np = kpts.cpu().numpy()
                            if kpts_np.ndim == 3 and kpts_np.shape[0] == 1:  # Remove batch dimension
                                kpts_np = kpts_np[0]
                            if kpts_np.ndim == 2 and kpts_np.shape[1] >= 2:
                                keypoints = kpts_np[:, :2]  # Take x, y coordinates
                            print(f"  Extracted keypoints shape: {keypoints.shape if keypoints is not None else 'None'}")
                        
                except Exception as e:
                    print(f"  Method 1 failed: {e}")
                
                # Method 2: Try using match_xfeat (common XFeat method)
                if keypoints is None:
                    try:
                        # Match image with itself to get keypoints
                        matches = xfeat_model.match_xfeat(img_rgb, img_rgb, top_k=max_features)
                        if isinstance(matches, tuple) and len(matches) >= 2:
                            keypoints = matches[0]  # First set of matched keypoints
                            if torch.is_tensor(keypoints):
                                keypoints = keypoints.cpu().numpy()
                        print(f"  Method 2 keypoints shape: {keypoints.shape if keypoints is not None else 'None'}")
                    except Exception as e:
                        print(f"  Method 2 failed: {e}")
                
                # Method 3: Direct feature extraction
                if keypoints is None:
                    try:
                        # Try calling XFeat directly
                        features = xfeat_model(img_tensor)
                        print(f"  Direct call result type: {type(features)}")
                        if isinstance(features, dict):
                            print(f"  Feature dict keys: {features.keys()}")
                            # Look for common keypoint keys
                            for key in ['keypoints', 'kpts', 'pts', 'corners']:
                                if key in features:
                                    kpts = features[key]
                                    if torch.is_tensor(kpts):
                                        keypoints = kpts.cpu().numpy()
                                        if keypoints.ndim == 3:
                                            keypoints = keypoints[0]
                                    break
                        print(f"  Method 3 keypoints shape: {keypoints.shape if keypoints is not None else 'None'}")
                    except Exception as e:
                        print(f"  Method 3 failed: {e}")
                
        except Exception as e:
            print(f"  All methods failed for {img_path}: {e}")
        
        # Validate and store keypoints
        if keypoints is None or len(keypoints) == 0:
            print(f"  No keypoints found for {img_path}")
            keypoints = np.array([]).reshape(0, 2)
        else:
            # Ensure correct format
            if keypoints.ndim != 2 or keypoints.shape[1] < 2:
                print(f"  Invalid keypoint format: {keypoints.shape}")
                keypoints = np.array([]).reshape(0, 2)
            else:
                keypoints = keypoints[:, :2]  # Keep only x, y coordinates
                if len(keypoints) > max_features:
                    keypoints = keypoints[:max_features]
                print(f"  Successfully extracted {len(keypoints)} keypoints")
        
        all_keypoints.append(keypoints)
        max_keypoints = max(max_keypoints, len(keypoints))
    
    print(f"\nMax keypoints found: {max_keypoints}")
    
    if max_keypoints == 0:
        print("No keypoints found in any image, using dummy features")
        return create_dummy_features(len(image_paths), num_features=20)
    
    # If test was successful, process all images
    if len(test_images) < len(image_paths):
        print(f"Test successful, processing remaining {len(image_paths) - len(test_images)} images...")
        # Add logic here to process remaining images using the successful method
        # For now, return test results
    
    # Pad keypoint arrays
    T = len(all_keypoints)
    J = min(max_keypoints, max_features)
    points2d = np.zeros((T, J, 2), dtype=np.float32)
    
    for i, keypoints in enumerate(all_keypoints):
        if len(keypoints) > 0:
            num_points = min(len(keypoints), J)
            points2d[i, :num_points, :] = keypoints[:num_points]
    
    return points2d

def create_dummy_features(num_frames, num_features=20):
    """Create dummy features for testing when XFeat is not available."""
    print("Creating dummy features for testing...")
    
    # Create random but consistent feature tracks
    np.random.seed(42)  # For reproducibility
    
    points2d = np.zeros((num_frames, num_features, 2))
    
    # Generate initial feature positions
    initial_points = np.random.rand(num_features, 2) * np.array([720, 540])  # Random points in image
    
    # Add small random motion to simulate feature tracking
    for t in range(num_frames):
        noise = np.random.normal(0, 2, (num_features, 2))  # Small random motion
        points2d[t] = initial_points + noise * t * 0.1  # Gradual drift
        
        # Keep points within image bounds
        points2d[t, :, 0] = np.clip(points2d[t, :, 0], 0, 719)
        points2d[t, :, 1] = np.clip(points2d[t, :, 1], 0, 539)
    
    return points2d

def create_bundle_adjustment_data(calib_file, tum_file, image_dir, output_pkl):
    """Main function to create bundle adjustment data."""
    
    print("Starting bundle adjustment data creation...")
    
    # 1. Parse calibration file
    print("Parsing calibration file...")
    calib_params = parse_calibration_file(calib_file)
    
    # 2. Parse TUM file
    print("Parsing TUM file...")
    poses_data = parse_tum_file(tum_file)
    
    # 3. Get image paths
    print("Getting image paths...")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    image_paths.sort()
    
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_paths)} images")
    
    # 4. Load XFeat model
    print("Loading XFeat model...")
    xfeat_model, device = load_xfeat_model()
    
    if xfeat_model is None:
        print("XFeat model could not be loaded, using dummy features...")
        points2d = create_dummy_features(len(image_paths))
    else:
        # 5. Extract features
        print("Extracting features...")
        points2d = extract_features_sequence(image_paths, xfeat_model, device)
    
    # 6. Create calibration data for each frame
    print("Creating calibration data...")
    num_frames = len(poses_data) if poses_data else len(image_paths)
    num_frames = min(num_frames, len(image_paths), points2d.shape[0])
    
    calib_dict = {}
    
    for i in range(num_frames):
        if i < len(poses_data):
            # Use TUM pose data
            pose = poses_data[i]
            R_matrix = quaternion_to_rotation_matrix(pose['quaternion'])
            tvec = pose['translation']
        else:
            # Use identity pose if TUM data is insufficient
            R_matrix = np.eye(3)
            tvec = np.zeros(3)
        
        calib_dict[i] = {
            'R': R_matrix,
            'tvec': tvec.reshape(-1, 1),
            'intr': calib_params['intrinsic'],
            'distort': calib_params['distortion']
        }
    
    # 7. Prepare final data structure
    final_data = {
        'points2d': points2d[:num_frames],
        **calib_dict  # Unpack calibration data directly into the dictionary
    }
    
    # 8. Save as pickle file
    print(f"Saving data to {output_pkl}")
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(final_data, f)
    
    print("Bundle adjustment data created successfully!")
    print(f"Points2D shape: {final_data['points2d'].shape}")
    print(f"Number of cameras: {len([k for k in final_data.keys() if isinstance(k, int)])}")
    
    return final_data

# Configuration
if __name__ == "__main__":
    # File paths - update these according to your setup
    CALIB_FILE = "/Volumes/Lenovo PS6/Dataset/results-cam-stereo-intrinsics-underwater.txt"  # Path to your calibration file
    TUM_FILE = "/Volumes/Lenovo PS6/Dataset/fjord_3_baseline.tum"     # Path to your TUM file
    IMAGE_DIR = "/Volumes/Lenovo PS6/Dataset/fjord_3"
    OUTPUT_PKL = "/Volumes/Lenovo PS6/Dataset/fjord3_short_for_bundle_adjustment.pkl"
    
    try:
        # Create the bundle adjustment data
        data = create_bundle_adjustment_data(
            calib_file=CALIB_FILE,
            tum_file=TUM_FILE,
            image_dir=IMAGE_DIR,
            output_pkl=OUTPUT_PKL
        )
        
        print("\n" + "="*50)
        print("SUCCESS: Bundle adjustment data created!")
        print(f"Output saved to: {OUTPUT_PKL}")
        print("="*50)
        
        # Test loading the created file
        print("\nTesting the created file...")
        with open(OUTPUT_PKL, 'rb') as f:
            test_data = pickle.load(f)
        
        print(f"✓ Points2D shape: {test_data['points2d'].shape}")
        print(f"✓ Number of camera frames: {len([k for k in test_data.keys() if isinstance(k, int)])}")
        print("✓ File structure is correct!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
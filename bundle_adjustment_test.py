import numpy as np
import cv2
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pathlib import Path
import os
from modules.xfeat import XFeat

class EquidistantCamera:
    """Equidistant distorted pinhole camera model."""
    
    def __init__(self, fx, fy, cx, cy, k1, k2, k3, k4):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k = np.array([k1, k2, k3, k4])  # Distortion coefficients
        
    def project(self, points_3d):
        """Project 3D points to image coordinates with equidistant distortion."""
        # Convert to camera coordinates
        x = points_3d[:, 0] / points_3d[:, 2]
        y = points_3d[:, 1] / points_3d[:, 2]
        
        # Calculate radius
        r = np.sqrt(x**2 + y**2)
        
        # Apply equidistant distortion
        theta = np.arctan(r)
        theta_d = theta * (1 + self.k[0] * theta**2 + self.k[1] * theta**4 + 
                          self.k[2] * theta**6 + self.k[3] * theta**8)
        
        # Avoid division by zero
        mask = r > 1e-8
        scale = np.ones_like(r)
        scale[mask] = theta_d[mask] / r[mask]
        
        x_d = x * scale
        y_d = y * scale
        
        # Apply intrinsics
        u = self.fx * x_d + self.cx
        v = self.fy * y_d + self.cy
        
        return np.column_stack([u, v])
    
    def unproject(self, image_points):
        """Unproject image points to normalized coordinates."""
        # Convert to normalized coordinates
        x_d = (image_points[:, 0] - self.cx) / self.fx
        y_d = (image_points[:, 1] - self.cy) / self.fy
        
        # Calculate distorted radius
        r_d = np.sqrt(x_d**2 + y_d**2)
        
        # Iteratively solve for undistorted angle
        theta = r_d.copy()
        for _ in range(10):  # Newton-Raphson iterations
            theta_d = theta * (1 + self.k[0] * theta**2 + self.k[1] * theta**4 + 
                              self.k[2] * theta**6 + self.k[3] * theta**8)
            f = theta_d - r_d
            df = (1 + 3 * self.k[0] * theta**2 + 5 * self.k[1] * theta**4 + 
                  7 * self.k[2] * theta**6 + 9 * self.k[3] * theta**8)
            theta = theta - f / (df + 1e-8)
        
        # Convert back to normalized coordinates
        mask = r_d > 1e-8
        scale = np.ones_like(r_d)
        scale[mask] = np.tan(theta[mask]) / r_d[mask]
        
        x = x_d * scale
        y = y_d * scale
        
        return np.column_stack([x, y])

class BundleAdjuster:
    """Bundle adjustment system for pose and structure optimization."""
    
    def __init__(self, camera):
        self.camera = camera
        self.poses = []  # List of [R, t] camera poses
        self.points_3d = []  # 3D point cloud
        self.observations = []  # List of 2D observations per image
        self.point_indices = []  # Which 3D points are observed in each image
        
    def load_poses_from_file(self, pose_file):
        """Load camera poses from text file."""
        poses = {}
        
        with open(pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                    
                image_name = parts[0]
                timestamp = float(parts[1])
                
                # Position
                t = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                
                # Quaternion (x, y, z, w)
                quat = np.array([float(parts[8]), float(parts[6]), 
                               float(parts[7]), float(parts[5])])  # Note: w is last in your format
                
                # Convert quaternion to rotation matrix
                rotation = R.from_quat(quat)
                R_matrix = rotation.as_matrix()
                
                poses[image_name] = {'R': R_matrix, 't': t, 'timestamp': timestamp}
                
        print(f"Loaded {len(poses)} poses from {pose_file}")
        if not poses:
            raise ValueError(f"No valid poses found in {pose_file}")
        
        return poses

    
    def extract_and_match_features(self, image_paths, xfeat_model, device, max_features=500):
        """Extract features and perform matching between consecutive images."""
        all_keypoints = []
        all_descriptors = []
        
        print("Extracting features from images...")
        
        for i, img_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            try:
                with torch.no_grad():
                    print(f"  Extracting features using XFeat...")
                    # Convert to tensor
                    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                    img_tensor = img_tensor.to(device)
                    
                    # Extract features using XFeat
                    outputs = xfeat_model.detectAndCompute(img_tensor, top_k=max_features)

                    outputs = outputs[0]

                    print(type(outputs))
                    
                    print(f"Features extracted: {len(outputs)}")
                    if len(outputs) == 3:
                        kpts = outputs['keypoints']
                        desc = outputs['descriptors']
                        print(f"  Keypoints: {kpts.shape}, Descriptors: {0.0}")
                        if torch.is_tensor(kpts):
                            kpts_np = kpts.cpu().numpy()
                            if kpts_np.ndim == 3 and kpts_np.shape[0] == 1:
                                kpts_np = kpts_np[0]
                            keypoints = kpts_np[:, :2] if kpts_np.shape[1] >= 2 else kpts_np
                            print(f"  Keypoints shape: {keypoints.shape}")
                        
                        if torch.is_tensor(desc):
                            descriptors = desc.cpu().numpy()
                            if descriptors.ndim == 3 and descriptors.shape[0] == 1:
                                descriptors = descriptors[0]
                                print(f"  Descriptors shape: {descriptors.shape}")
                        
                        all_keypoints.append(keypoints)
                        all_descriptors.append(descriptors)
                        
                        print(f"  Extracted {len(keypoints)} keypoints")
                    else:
                        print(f"  Failed to extract features from {img_path}")
                        all_keypoints.append(np.array([]).reshape(0, 2))
                        all_descriptors.append(np.array([]).reshape(0, -1))
                        
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                all_keypoints.append(np.array([]).reshape(0, 2))
                all_descriptors.append(np.array([]).reshape(0, -1))
        
        return all_keypoints, all_descriptors
    
    def match_features(self, desc1, desc2, ratio_threshold=0.8):
        """Match features between two images using descriptor similarity."""
        if len(desc1) == 0 or len(desc2) == 0:
            return np.array([]).reshape(0, 2)
        
        # Compute pairwise distances
        distances = np.linalg.norm(desc1[:, None] - desc2[None, :], axis=2)
        
        # Find best matches
        best_matches = np.argmin(distances, axis=1)
        best_distances = np.min(distances, axis=1)
        
        # Find second best matches for ratio test
        distances_copy = distances.copy()
        distances_copy[np.arange(len(desc1)), best_matches] = np.inf
        second_best_distances = np.min(distances_copy, axis=1)
        
        # Apply ratio test
        ratio = best_distances / (second_best_distances + 1e-8)
        good_matches = ratio < ratio_threshold
        
        # Create match indices
        matches = []
        for i in range(len(desc1)):
            if good_matches[i]:
                matches.append([i, best_matches[i]])
        
        return np.array(matches) if matches else np.array([]).reshape(0, 2)
    
    def triangulate_points(self, kpts1, kpts2, R1, t1, R2, t2):
        """Triangulate 3D points from matched 2D points."""
        if len(kpts1) == 0 or len(kpts2) == 0:
            return np.array([]).reshape(0, 3)
        
        # Create projection matrices
        P1 = np.hstack([R1, t1.reshape(-1, 1)])
        P2 = np.hstack([R2, t2.reshape(-1, 1)])
        
        # Add camera intrinsics
        K = np.array([[self.camera.fx, 0, self.camera.cx],
                     [0, self.camera.fy, self.camera.cy],
                     [0, 0, 1]])
        
        P1 = K @ P1
        P2 = K @ P2
        
        points_3d = []
        
        for i in range(len(kpts1)):
            # Triangulate using DLT
            A = np.array([
                kpts1[i, 0] * P1[2, :] - P1[0, :],
                kpts1[i, 1] * P1[2, :] - P1[1, :],
                kpts2[i, 0] * P2[2, :] - P2[0, :],
                kpts2[i, 1] * P2[2, :] - P2[1, :]
            ])
            
            _, _, Vt = np.linalg.svd(A)
            point_3d = Vt[-1, :3] / Vt[-1, 3]  # Homogeneous to 3D
            points_3d.append(point_3d)
        
        return np.array(points_3d)
    
    def setup_bundle_adjustment(self, image_paths, poses_dict, all_keypoints, all_descriptors):
        """Set up the bundle adjustment problem."""
        print("Setting up bundle adjustment...")
        
        # Initialize poses
        self.poses = []
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            if img_name in poses_dict:
                pose = poses_dict[img_name]
                self.poses.append([pose['R'], pose['t']])
            else:
                print(f"Warning: No pose found for {img_name}")
                # Use identity pose as fallback
                self.poses.append([np.eye(3), np.zeros(3)])
        
        # Match features between consecutive images and triangulate
        all_matches = []
        self.observations = [[] for _ in range(len(image_paths))]
        self.point_indices = [[] for _ in range(len(image_paths))]
        
        point_id = 0
        
        for i in range(len(image_paths) - 1):
            print(f"Matching features between image {i} and {i+1}")
            
            # Match features
            matches = self.match_features(all_descriptors[i], all_descriptors[i+1])
            
            if len(matches) > 0:
                # Get matched keypoints
                kpts1 = all_keypoints[i][matches[:, 0]]
                kpts2 = all_keypoints[i+1][matches[:, 1]]
                
                # Triangulate 3D points
                points_3d = self.triangulate_points(
                    kpts1, kpts2,
                    self.poses[i][0], self.poses[i][1],
                    self.poses[i+1][0], self.poses[i+1][1]
                )
                
                # Filter valid points (positive depth, reasonable distance)
                valid_mask = (points_3d[:, 2] > 0.1) & (np.linalg.norm(points_3d, axis=1) < 100)
                valid_points = points_3d[valid_mask]
                valid_kpts1 = kpts1[valid_mask]
                valid_kpts2 = kpts2[valid_mask]
                
                # Add to observations
                for j, (pt3d, kpt1, kpt2) in enumerate(zip(valid_points, valid_kpts1, valid_kpts2)):
                    self.points_3d.append(pt3d)
                    
                    # Add observations for both images
                    self.observations[i].append(kpt1)
                    self.point_indices[i].append(point_id)
                    
                    self.observations[i+1].append(kpt2)
                    self.point_indices[i+1].append(point_id)
                    
                    point_id += 1
                
                print(f"  Found {len(valid_points)} valid matches")
        
        # Convert to numpy arrays
        self.points_3d = np.array(self.points_3d)
        for i in range(len(self.observations)):
            if self.observations[i]:
                self.observations[i] = np.array(self.observations[i])
            else:
                self.observations[i] = np.array([]).reshape(0, 2)
        
        print(f"Bundle adjustment setup complete:")
        print(f"  - {len(self.poses)} camera poses")
        print(f"  - {len(self.points_3d)} 3D points")
        print(f"  - Total observations: {sum(len(obs) for obs in self.observations)}")
    
    def residual_function(self, params):
        """Compute reprojection residuals for bundle adjustment."""
        n_cameras = len(self.poses)
        n_points = len(self.points_3d)
        
        # Extract camera parameters (6 DOF per camera: 3 rotation + 3 translation)
        camera_params = params[:n_cameras * 6].reshape(n_cameras, 6)
        
        # Extract 3D points
        points_3d = params[n_cameras * 6:].reshape(n_points, 3)
        
        residuals = []
        
        for cam_idx in range(n_cameras):
            if len(self.observations[cam_idx]) == 0:
                continue
            
            # Convert rotation vector to matrix
            rvec = camera_params[cam_idx, :3]
            tvec = camera_params[cam_idx, 3:6]
            R_mat = cv2.Rodrigues(rvec)[0]
            
            # Get observed points for this camera
            observed_points = self.observations[cam_idx]
            point_ids = self.point_indices[cam_idx]
            
            # Transform 3D points to camera coordinates
            points_cam = (R_mat @ points_3d[point_ids].T + tvec.reshape(-1, 1)).T
            
            # Project to image coordinates
            projected = self.camera.project(points_cam)
            
            # Compute residuals
            residual = (observed_points - projected).flatten()
            residuals.extend(residual)
        
        return np.array(residuals)
    
    def optimize(self, max_iterations=100):
        """Perform bundle adjustment optimization."""
        if len(self.points_3d) == 0:
            print("No 3D points to optimize!")
            return
        
        print("Starting bundle adjustment optimization...")
        
        n_cameras = len(self.poses)
        n_points = len(self.points_3d)
        
        # Initialize parameters
        camera_params = []
        for R, t in self.poses:
            # Convert rotation matrix to rodrigues vector
            rvec = cv2.Rodrigues(R)[0].flatten()
            camera_params.extend(rvec)
            camera_params.extend(t)
        
        # Combine camera and point parameters
        initial_params = np.concatenate([camera_params, self.points_3d.flatten()])
        
        print(f"Optimizing {n_cameras} cameras and {n_points} points...")
        print(f"Total parameters: {len(initial_params)}")
        
        # Run optimization
        result = least_squares(
            self.residual_function,
            initial_params,
            method='lm',
            max_nfev=max_iterations * len(initial_params),
            verbose=1
        )
        
        # Extract optimized parameters
        optimized_params = result.x
        optimized_camera_params = optimized_params[:n_cameras * 6].reshape(n_cameras, 6)
        optimized_points = optimized_params[n_cameras * 6:].reshape(n_points, 3)
        
        # Update poses and points
        for i in range(n_cameras):
            rvec = optimized_camera_params[i, :3]
            tvec = optimized_camera_params[i, 3:6]
            R_mat = cv2.Rodrigues(rvec)[0]
            self.poses[i] = [R_mat, tvec]
        
        self.points_3d = optimized_points
        
        print(f"Optimization completed!")
        print(f"Final cost: {result.cost:.6f}")
        print(f"Iterations: {result.nfev}")
        
        return result
    
    def visualize_results(self):
        """Visualize the bundle adjustment results."""
        if len(self.points_3d) == 0:
            print("No 3D points to visualize!")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D points
        ax.scatter(self.points_3d[:, 0], self.points_3d[:, 1], self.points_3d[:, 2], 
                  c='blue', alpha=0.6, s=1, label='3D Points')
        
        # Plot camera positions
        camera_positions = np.array([t for R, t in self.poses])
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                  c='red', s=100, marker='^', label='Cameras')
        
        # Draw camera coordinate frames
        for i, (R, t) in enumerate(self.poses):
            # Draw coordinate axes
            axes = R.T * 0.5  # Scale for visibility
            colors = ['r', 'g', 'b']
            for j, color in enumerate(colors):
                ax.plot([t[0], t[0] + axes[0, j]], 
                       [t[1], t[1] + axes[1, j]], 
                       [t[2], t[2] + axes[2, j]], color=color, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Bundle Adjustment Results')
        
        plt.tight_layout()
        plt.show()

    def load_xfeat_model(self,device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Load XFeat model for inference."""
        try:
            xfeat = XFeat()
            xfeat.to(device)
            xfeat.eval()
            return xfeat, device
        except Exception as e:
            print(f"Error loading XFeat model: {e}")
            return None, device

def main():
    """Main function to run bundle adjustment."""
    
    # Camera parameters from your file
    camera = EquidistantCamera(
        fx=460.11634467, fy=460.43729775,
        cx=357.88255738, cy=266.4372886,
        k1=0.04816514, k2=0.170736, k3=-0.30080163, k4=0.47220233
    )
    
    # Initialize bundle adjuster
    ba = BundleAdjuster(camera)
    
    # Example usage:
    # 1. Load your XFeat model (you'll need to replace this with your actual model loading)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xfeat_model, device = ba.load_xfeat_model(device)  # Your XFeat model loading code
    
    # 2. Set up your image paths and pose file
    image_paths = ["/Volumes/Lenovo PS6/Dataset/cam_0/image_000260.jpg", "/Volumes/Lenovo PS6/Dataset/cam_0/image_000266.jpg"]
    pose_file = "/Volumes/Lenovo PS6/Dataset/test_data_ba.txt"
    
    # 3. Load poses and extract features
    poses_dict = ba.load_poses_from_file(pose_file)
    all_keypoints, all_descriptors = ba.extract_and_match_features(
        image_paths, xfeat_model, device
    )
    
    # 4. Set up and run bundle adjustment
    ba.setup_bundle_adjustment(image_paths, poses_dict, all_keypoints, all_descriptors)
    result = ba.optimize()
    print(f"Optimization result: {result}")

    # 5. Visualize results
    ba.visualize_results()
    
    print("Bundle adjustment system ready!")
    print("To use:")
    print("1. Load your XFeat model")
    print("2. Set up image paths and pose file")
    print("3. Run the pipeline as shown in the main() function")

if __name__ == "__main__":
    main()
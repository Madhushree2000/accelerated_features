import numpy as np
import cv2
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
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
        self.image_names = []  # Store image names for reference
        
    def load_poses_from_file(self, pose_file, quat_format='auto'):
        """
        Load camera poses from text file with automatic quaternion format detection.
        
        Args:
            pose_file: Path to pose file
            quat_format: 'auto', 'wxyz', 'xyzw', or 'custom'
        """
        poses = {}
        
        with open(pose_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            raise ValueError(f"No valid lines found in {pose_file}")
        
        # Parse first few poses to determine format
        test_poses = []
        for line in lines[:min(3, len(lines))]:
            parts = line.split()
            if len(parts) < 9:
                continue
                
            image_name = parts[0]
            timestamp = float(parts[1])
            t = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            
            # Try different quaternion formats
            formats = {
                'wxyz': np.array([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])]),
                'xyzw': np.array([float(parts[8]), float(parts[6]), float(parts[7]), float(parts[5])]),
            }
            
            test_poses.append({
                'name': image_name,
                'timestamp': timestamp,
                't': t,
                'formats': formats
            })
        
        # Auto-detect best format if requested
        if quat_format == 'auto':
            quat_format = self.detect_quaternion_format(test_poses)
            print(f"Auto-detected quaternion format: {quat_format}")
        
        # Load all poses with determined format
        for line in lines:
            parts = line.split()
            if len(parts) < 9:
                continue
                
            image_name = parts[0]
            timestamp = float(parts[1])
            t = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            
            if quat_format == 'wxyz':
                quat = np.array([float(parts[6]), float(parts[7]), float(parts[8]), float(parts[5])])  # Convert to [x,y,z,w]
            elif quat_format == 'xyzw':
                quat = np.array([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])  # Already [x,y,z,w]
            else:
                raise ValueError(f"Unknown quaternion format: {quat_format}")
            
            # Validate quaternion
            quat_norm = np.linalg.norm(quat)
            if abs(quat_norm - 1.0) > 0.1:
                print(f"Warning: Quaternion not normalized for {image_name}: norm = {quat_norm}")
                quat = quat / quat_norm
            
            # Convert to rotation matrix
            try:
                rotation = R.from_quat(quat)  # scipy expects [x,y,z,w]
                R_matrix = rotation.as_matrix()
                
                # Validate rotation matrix
                if not self.is_valid_rotation_matrix(R_matrix):
                    print(f"Warning: Invalid rotation matrix for {image_name}")
                    continue
                    
                poses[image_name] = {'R': R_matrix, 't': t, 'timestamp': timestamp}
                
            except Exception as e:
                print(f"Error processing pose for {image_name}: {e}")
                continue
        
        print(f"Loaded {len(poses)} valid poses from {pose_file}")
        if len(poses) == 0:
            raise ValueError(f"No valid poses found in {pose_file}")
        
        return poses

    def detect_quaternion_format(self, test_poses):
        """Detect quaternion format by testing geometric consistency."""
        formats = ['wxyz', 'xyzw']
        best_format = 'wxyz'
        best_score = -1
        
        for fmt in formats:
            score = 0
            valid_poses = 0
            
            for pose_data in test_poses:
                try:
                    quat = pose_data['formats'][fmt] if fmt in pose_data['formats'] else None
                    if quat is None:
                        continue
                    
                    # Normalize quaternion
                    quat_norm = np.linalg.norm(quat)
                    if quat_norm > 0:
                        quat = quat / quat_norm
                    else:
                        continue
                    
                    # Convert to rotation matrix (scipy expects [x,y,z,w])
                    if fmt == 'wxyz':
                        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])  # w,x,y,z -> x,y,z,w
                    else:
                        quat_scipy = quat  # Already x,y,z,w
                    
                    rotation = R.from_quat(quat_scipy)
                    R_matrix = rotation.as_matrix()
                    
                    # Check if rotation matrix is valid
                    if self.is_valid_rotation_matrix(R_matrix):
                        score += 1
                        valid_poses += 1
                        
                except Exception:
                    continue
            
            print(f"Format {fmt}: {valid_poses}/{len(test_poses)} valid poses")
            
            if valid_poses > best_score:
                best_score = valid_poses
                best_format = fmt
        
        return best_format

    def is_valid_rotation_matrix(self, R, tolerance=1e-6):
        """Check if matrix is a valid rotation matrix."""
        if R.shape != (3, 3):
            return False
        
        # Check if orthogonal: R @ R.T = I
        should_be_identity = np.dot(R, R.T)
        identity = np.eye(3)
        if not np.allclose(should_be_identity, identity, atol=tolerance):
            return False
        
        # Check if determinant is 1 (not -1, which would be reflection)
        if not np.isclose(np.linalg.det(R), 1.0, atol=tolerance):
            return False
        
        return True

    def validate_poses_geometry(self, poses_dict):
        """Validate pose geometry and detect potential issues."""
        pose_list = list(poses_dict.values())
        
        if len(pose_list) < 2:
            return True
        
        # Check camera translations
        positions = np.array([pose['t'] for pose in pose_list])
        distances = []
        
        for i in range(len(positions) - 1):
            dist = np.linalg.norm(positions[i+1] - positions[i])
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        print(f"Average camera movement: {avg_distance:.3f}")
        
        # Check for unreasonable movements
        if avg_distance > 10.0:
            print("Warning: Large camera movements detected - check pose units")
        elif avg_distance < 0.001:
            print("Warning: Very small camera movements - check pose precision")
        
        # Check rotation consistency
        rotations = [pose['R'] for pose in pose_list]
        for i, R in enumerate(rotations):
            if not self.is_valid_rotation_matrix(R):
                print(f"Warning: Invalid rotation matrix for pose {i}")
                return False
        
        return True

    def get_image_list(self, image_folder, poses_dict):
        """Get list of images that have corresponding poses."""
        # Get all image files in the folder
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
            image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
        
        # Filter images that have corresponding poses
        valid_images = []
        for img_path in sorted(image_files):
            img_name = os.path.basename(img_path)
            if img_name in poses_dict:
                valid_images.append(img_path)
        
        print(f"Found {len(image_files)} total images")
        print(f"Found {len(valid_images)} images with corresponding poses")
        
        if len(valid_images) < 2:
            raise ValueError("Need at least 2 images with poses for bundle adjustment")
        
        return valid_images
    
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

                    print(f"Features extracted: {len(outputs)}")
                    if len(outputs) == 3:
                        kpts = outputs['keypoints']
                        desc = outputs['descriptors']
                        print(f"  Keypoints: {kpts.shape}, Descriptors: {desc.shape}")
                        
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
    
    def setup_bundle_adjustment_multiple_images(self, image_paths, poses_dict, all_keypoints, all_descriptors):
        """Set up bundle adjustment for multiple images with better feature tracking."""
        print("Setting up bundle adjustment for multiple images...")
        
        # Store image names
        self.image_names = [os.path.basename(path) for path in image_paths]
        
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
        
        # Initialize observation lists
        self.observations = [[] for _ in range(len(image_paths))]
        self.point_indices = [[] for _ in range(len(image_paths))]
        self.points_3d = []
        
        # Track features across multiple images
        point_tracks = {}  # Dictionary to store point tracks
        point_id = 0
        
        # Match features between consecutive images
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
                
                # Create point tracks
                for j, (pt3d, kpt1, kpt2) in enumerate(zip(valid_points, valid_kpts1, valid_kpts2)):
                    track_key = f"{i}_{i+1}_{j}"  # Unique identifier for this track
                    
                    # Check if this point can be linked to existing tracks
                    # (simplified approach - in practice, you'd want more sophisticated track linking)
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
        
        # Print observation statistics per image
        for i, obs in enumerate(self.observations):
            print(f"  - Image {i} ({self.image_names[i]}): {len(obs)} observations")
    
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
    
    def optimize(self, max_iterations=200):
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
        
        # Calculate initial residuals
        initial_residuals = self.residual_function(initial_params)
        initial_cost = 0.5 * np.sum(initial_residuals**2)
        print(f"Initial cost: {initial_cost:.6f}")
        
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
            print(f"Camera {i} completed")
        
        self.points_3d = optimized_points
        
        print(f"Optimization completed!")
        print(f"Initial cost: {initial_cost:.6f}")
        print(f"Final cost: {result.cost:.6f}")
        print(f"Cost reduction: {((initial_cost - result.cost) / initial_cost * 100):.2f}%")
        print(f"Iterations: {result.nfev}")
        
        return result
    
    def visualize_results(self):
        """Visualize the bundle adjustment results."""
        if len(self.points_3d) == 0:
            print("No 3D points to visualize!")
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D visualization
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot 3D points
        ax1.scatter(self.points_3d[:, 0], self.points_3d[:, 1], self.points_3d[:, 2], 
                   c='blue', alpha=0.6, s=1, label='3D Points')
        
        # Plot camera positions
        camera_positions = np.array([t for R, t in self.poses])
        ax1.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                   c='red', s=100, marker='^', label='Cameras')
        
        # Draw camera coordinate frames
        for i, (R, t) in enumerate(self.poses):
            # Draw coordinate axes
            axes = R.T * 0.5  # Scale for visibility
            colors = ['r', 'g', 'b']
            for j, color in enumerate(colors):
                ax1.plot([t[0], t[0] + axes[0, j]], 
                        [t[1], t[1] + axes[1, j]], 
                        [t[2], t[2] + axes[2, j]], color=color, alpha=0.7)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        ax1.set_title('Bundle Adjustment Results - 3D View')
        
        # Top-down view
        ax2 = fig.add_subplot(122)
        ax2.scatter(self.points_3d[:, 0], self.points_3d[:, 1], 
                   c='blue', alpha=0.6, s=1, label='3D Points')
        ax2.scatter(camera_positions[:, 0], camera_positions[:, 1],
                   c='red', s=100, marker='^', label='Cameras')
        
        # Draw camera trajectory
        ax2.plot(camera_positions[:, 0], camera_positions[:, 1], 
                'r-', alpha=0.5, linewidth=2, label='Camera Path')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        ax2.set_title('Bundle Adjustment Results - Top View')
        ax2.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"Number of cameras: {len(self.poses)}")
        print(f"Number of 3D points: {len(self.points_3d)}")
        print(f"Camera trajectory length: {np.sum(np.linalg.norm(np.diff(camera_positions, axis=0), axis=1)):.2f}")
        print(f"Point cloud extent:")
        print(f"  X: {self.points_3d[:, 0].min():.2f} to {self.points_3d[:, 0].max():.2f}")
        print(f"  Y: {self.points_3d[:, 1].min():.2f} to {self.points_3d[:, 1].max():.2f}")
        print(f"  Z: {self.points_3d[:, 2].min():.2f} to {self.points_3d[:, 2].max():.2f}")

    def load_xfeat_model(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
    """Main function to run bundle adjustment on image folder."""
    
    # Camera parameters from your file
    camera = EquidistantCamera(
        fx=460.11634467, fy=460.43729775,
        cx=357.88255738, cy=266.4372886,
        k1=0.04816514, k2=0.170736, k3=-0.30080163, k4=0.47220233
    )
    
    # Initialize bundle adjuster
    ba = BundleAdjuster(camera)
    
    # Set up paths
    image_folder = "/Volumes/Lenovo PS6/Dataset/Test_Images2"
    pose_file = "/Volumes/Lenovo PS6/Dataset/Test_Images2/selected_poses.txt"
    
    # Load XFeat model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    xfeat_model, device = ba.load_xfeat_model(device)
    
    if xfeat_model is None:
        print("Failed to load XFeat model. Exiting.")
        return
    
    try:
        # Load poses
        print("Loading poses...")
        poses_dict = ba.load_poses_from_file(pose_file)
        
        # Get image list
        print("Getting image list...")
        image_paths = ba.get_image_list(image_folder, poses_dict)
        
        # Limit number of images for testing (remove this line to process all images)
        # image_paths = image_paths[:10]  # Process first 10 images only
        
        print(f"Processing {len(image_paths)} images...")
        
        # Extract features from all images
        print("Extracting features...")
        all_keypoints, all_descriptors = ba.extract_and_match_features(
            image_paths, xfeat_model, device, max_features=1000
        )
        
        # Set up and run bundle adjustment
        print("Setting up bundle adjustment...")
        ba.setup_bundle_adjustment_multiple_images(image_paths, poses_dict, all_keypoints, all_descriptors)
        
        if len(ba.points_3d) > 0:
            print("Running optimization...")
            result = ba.optimize(max_iterations=100)
            
            # Visualize results
            print("Visualizing results...")
            ba.visualize_results()
            
            print("Bundle adjustment completed successfully!")
        else:
            print("No 3D points found. Check feature matching and pose data.")
    
    except Exception as e:
        print(f"Error during bundle adjustment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
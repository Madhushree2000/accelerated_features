import os
import numpy as np
import cv2
import ceres
from ceres import LossFunction, Problem, Solver
from ceres import CauchyLoss, HuberLoss

def load_camera_data(camera_data_dict):
    """Convert camera data dictionary into usable format"""
    cameras = {}
    for img_id, data in camera_data_dict.items():
        cameras[img_id] = {
            'R': data['R'],
            'tvec': data['tvec'],
            'intr': data['intr'],
            'distort': data['distort']
        }
    return cameras

def load_keypoints(keypoints_data):
    """Convert keypoints data into usable format"""
    # Assuming keypoints_data is T x J x 2 numpy array
    # T = number of frames, J = number of points per frame
    return keypoints_data

def project_point(point_3d, camera_params, intrinsics, dist_coeffs):
    """Project 3D point to 2D using camera parameters"""
    # Convert camera params to rotation vector and translation
    rvec = cv2.Rodrigues(camera_params[:3])[0]
    tvec = camera_params[3:6]
    
    # Project point
    point_2d, _ = cv2.projectPoints(
        point_3d.reshape(1, 1, 3),
        rvec,
        tvec,
        intrinsics,
        dist_coeffs
    )
    return point_2d.reshape(2)

class BundleAdjustmentProblem:
    def __init__(self, cameras, keypoints, intrinsics, dist_coeffs):
        self.cameras = cameras
        self.keypoints = keypoints
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
        self.problem = Problem()
        
        # Camera parameter dimension (rotation vector + translation)
        self.camera_param_size = 6
        
        # Point dimension
        self.point_param_size = 3
        
        # Create parameter blocks
        self.setup_parameters()
        
        # Add residuals
        self.add_residuals()
    
    def setup_parameters(self):
        # Initialize camera parameters
        self.camera_params = {}
        for img_id, cam_data in self.cameras.items():
            # Convert rotation matrix to rotation vector
            rvec = cv2.Rodrigues(cam_data['R'])[0].flatten()
            tvec = cam_data['tvec'].flatten()
            
            # Combine into single parameter vector
            params = np.concatenate([rvec, tvec])
            self.camera_params[img_id] = params
            
            # Add to problem
            self.problem.AddParameterBlock(params, self.camera_param_size)
        
        # Initialize 3D points (landmarks)
        num_points = self.keypoints.shape[1]
        self.points_3d = np.random.randn(num_points, 3)  # Initial guess
        
        for i in range(num_points):
            self.problem.AddParameterBlock(self.points_3d[i], self.point_param_size)
    
    def add_residuals(self):
        # Create loss function (Huber loss to reduce influence of outliers)
        loss_function = HuberLoss(1.0)
        
        # Add residuals for each observation
        for img_id in self.cameras.keys():
            img_idx = int(img_id)  # Assuming img_id can be converted to int
            camera_param = self.camera_params[img_id]
            
            for point_idx in range(self.keypoints.shape[1]):
                observed_point = self.keypoints[img_idx, point_idx]
                
                # Skip if point is not observed (assuming NaN or similar)
                if np.isnan(observed_point).any():
                    continue
                
                # Add residual block
                self.problem.AddResidualBlock(
                    ReprojectionError.create(
                        observed_point,
                        self.intrinsics,
                        self.dist_coeffs
                    ),
                    loss_function,
                    camera_param,
                    self.points_3d[point_idx]
                )
    
    def solve(self):
        options = Solver.Options()
        options.linear_solver_type = ceres.DENSE_SCHUR
        options.minimizer_progress_to_stdout = True
        options.max_num_iterations = 100
        options.function_tolerance = 1e-6
        
        summary = Solver.Summary()
        self.problem.Solve(options, summary)
        
        print(summary.BriefReport())
        
        # Update camera parameters with optimized values
        for img_id in self.cameras.keys():
            params = self.camera_params[img_id]
            rvec = params[:3]
            tvec = params[3:6]
            
            # Convert rotation vector back to matrix
            R, _ = cv2.Rodrigues(rvec)
            
            self.cameras[img_id]['R'] = R
            self.cameras[img_id]['tvec'] = tvec.reshape(3, 1)
        
        return self.cameras, self.points_3d

class ReprojectionError:
    def __init__(self, observed_point, intrinsics, dist_coeffs):
        self.observed_point = observed_point
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
    
    @staticmethod
    def create(observed_point, intrinsics, dist_coeffs):
        return ReprojectionError(observed_point, intrinsics, dist_coeffs)
    
    def __call__(self, camera_params, point_3d, residuals):
        # Extract rotation and translation
        rvec = camera_params[:3]
        tvec = camera_params[3:6]
        
        # Project 3D point
        projected_point, _ = cv2.projectPoints(
            point_3d.reshape(1, 1, 3),
            rvec,
            tvec,
            self.intrinsics,
            self.dist_coeffs
        )
        projected_point = projected_point.reshape(2)
        
        # Compute residuals
        residuals[0] = projected_point[0] - self.observed_point[0]
        residuals[1] = projected_point[1] - self.observed_point[1]
        
        return True

def bundle_adjustment(camera_data, keypoints_data):
    # Load data
    cameras = load_camera_data(camera_data)
    keypoints = load_keypoints(keypoints_data)
    
    # Use intrinsics and distortion from first camera (assuming same for all)
    first_cam = cameras[next(iter(cameras))]
    intrinsics = first_cam['intr']
    dist_coeffs = first_cam['distort']
    
    # Create and solve bundle adjustment problem
    ba_problem = BundleAdjustmentProblem(cameras, keypoints, intrinsics, dist_coeffs)
    optimized_cameras, optimized_points = ba_problem.solve()
    
    return optimized_cameras, optimized_points

# Example usage
if __name__ == "__main__":
    # Your camera data dictionary
    camera_data = {
        0: {
            'R': np.array([[ 0.52291726, -0.85113411, -0.04613302],
                          [-0.83258669, -0.5216205 , 0.18631013],
                          [-0.18263884, -0.05901505, -0.9814073 ]]),
            'tvec': np.array([[-1.042841], [-1.220057], [ 1.130509]]),
            'intr': np.array([[460.11634467, 0., 357.88255738],
                            [0., 460.43729775, 266.4372886],
                            [0., 0., 1.]]),
            'distort': np.array([0.04816514, 0.170736, -0.30080163, 0.47220233])
        },
        # Add more cameras as needed
    }
    
    # Your keypoints data (T x J x 2 numpy array)
    keypoints_data = np.random.randn(10, 100, 2)  # Replace with your actual data
    
    # Run bundle adjustment
    optimized_cameras, optimized_points = bundle_adjustment(camera_data, keypoints_data)
    
    print("Optimization complete!")
    print("Optimized camera parameters:", optimized_cameras)
    print("Optimized 3D points shape:", optimized_points.shape)
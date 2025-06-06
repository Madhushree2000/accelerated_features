import os
import shutil
from pathlib import Path

def extract_images_and_poses(source_folder, pose_file, output_folder, output_pose_file):
    """
    Extract every 10th image from image_000000.jpg to image_002000.jpg
    and create corresponding pose file.
    
    Args:
        source_folder: Path to folder containing original images
        pose_file: Path to original pose .txt file
        output_folder: Path to output folder for selected images
        output_pose_file: Path to output pose .txt file
    """
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Read original pose file
    poses = {}
    try:
        with open(pose_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split()
                    if len(parts) >= 8:  # Ensure we have all required fields
                        image_name = parts[0]
                        poses[image_name] = line
    except FileNotFoundError:
        print(f"Error: Pose file '{pose_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading pose file: {e}")
        return
    
    # Generate list of images to extract (every 10th from 0 to 2000)
    selected_images = []
    copied_count = 0
    pose_entries = []
    
    for i in range(62, 108, 10):  # 0, 10, 20, ..., 2000
        image_name = f"image_{i:06d}.jpg"
        source_path = os.path.join(source_folder, image_name)
        
        # Check if image exists in source folder
        if os.path.exists(source_path):
            # Copy image to output folder
            destination_path = os.path.join(output_folder, image_name)
            try:
                shutil.copy2(source_path, destination_path)
                selected_images.append(image_name)
                copied_count += 1
                print(f"Copied: {image_name}")
                
                # Get corresponding pose data
                if image_name in poses:
                    pose_entries.append(poses[image_name])
                else:
                    print(f"Warning: No pose data found for {image_name}")
                    
            except Exception as e:
                print(f"Error copying {image_name}: {e}")
        else:
            print(f"Warning: {image_name} not found in source folder")
    
    # Write filtered pose file
    try:
        with open(output_pose_file, 'w') as f:
            for pose_entry in pose_entries:
                f.write(pose_entry + '\n')
        print(f"\nCreated pose file: {output_pose_file}")
        print(f"Pose entries written: {len(pose_entries)}")
    except Exception as e:
        print(f"Error writing pose file: {e}")
    
    print(f"\nSummary:")
    print(f"Images copied: {copied_count}")
    print(f"Images selected: {len(selected_images)}")
    print(f"Pose entries: {len(pose_entries)}")

# Example usage
if __name__ == "__main__":
    # Configure these paths according to your setup
    SOURCE_FOLDER = "/Volumes/Lenovo PS6/Dataset/mclab_2"  # Replace with your image folder path
    POSE_FILE = "/Volumes/Lenovo PS6/Dataset/mclab_2/image_pose_data.txt"         # Replace with your pose file path
    OUTPUT_FOLDER = "/Volumes/Lenovo PS6/Dataset/Test_Images2"            # Output folder for selected images
    OUTPUT_POSE_FILE = "/Volumes/Lenovo PS6/Dataset/Test_Images2/selected_poses.txt"      # Output pose file
    
    print("Starting image and pose extraction...")
    print(f"Source folder: {SOURCE_FOLDER}")
    print(f"Pose file: {POSE_FILE}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Output pose file: {OUTPUT_POSE_FILE}")
    print("-" * 50)
    
    extract_images_and_poses(
        source_folder=SOURCE_FOLDER,
        pose_file=POSE_FILE,
        output_folder=OUTPUT_FOLDER,
        output_pose_file=OUTPUT_POSE_FILE
    )
    
    print("\nExtraction complete!")
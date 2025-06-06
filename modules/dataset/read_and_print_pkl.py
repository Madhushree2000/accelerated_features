import pickle
import os

import pickle
import os
import numpy as np

def read_and_save_pkl_contents(filepath, output_txt_path):
    """
    Reads a .pkl file and saves its content to a .txt file.
    WARNING: Only unpickle data from trusted sources!
    """
    if not os.path.exists(filepath):
        with open(output_txt_path, 'w') as txt_file:
            txt_file.write(f"Error: The file '{filepath}' was not found.")
        return

    try:
        with open(filepath, 'rb') as file:
            # Configure NumPy to print full arrays
            np.set_printoptions(threshold=np.inf)
            
            data = pickle.load(file)
            
            with open(output_txt_path, 'w') as txt_file:
                txt_file.write(f"--- Contents of '{filepath}' ---\n")
                txt_file.write(str(data))
                
                # Write the type of the loaded object
                txt_file.write(f"\n\n--- Type of loaded data: {type(data)} ---")

    except pickle.UnpicklingError as e:
        with open(output_txt_path, 'w') as txt_file:
            txt_file.write(f"Error unpickling file '{filepath}': {e}\n")
            txt_file.write("This often means the file is corrupted or not a valid pickle file.")
    except Exception as e:
        with open(output_txt_path, 'w') as txt_file:
            txt_file.write(f"An unexpected error occurred: {e}")
    finally:
        # Reset print options to default
        np.set_printoptions(threshold=1000)  # Default threshold value

if __name__ == "__main__":
    # Path to the pickle file
    pkl_file_path = "/Users/madhushreesannigrahi/Downloads/pose_result__home_user_Desktop_DeepFly3D_data_test.pkl"
    
    # Path for the output text file (you can modify this as needed)
    output_txt_path = os.path.join(os.path.dirname(pkl_file_path), "pkl_contents.txt")
    
    read_and_save_pkl_contents(pkl_file_path, output_txt_path)
    print(f"Contents saved to: {output_txt_path}")
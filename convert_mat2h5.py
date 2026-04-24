import os
import h5py
import mat73
import numpy as np

def save_dict_to_hdf5(group, data_dict):
    """Recursively saves a dictionary to an H5 group."""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Create a subgroup for nested dictionaries/MATLAB structures
            subgroup = group.create_group(key)
            save_dict_to_hdf5(subgroup, value)
        elif isinstance(value, np.ndarray):
            # Save arrays (including multi-dimensional ones like y0all)
            group.create_dataset(key, data=value)
        else:
            # Handle scalars or strings
            try:
                group.create_dataset(key, data=value)
            except TypeError:
                # Fallback for complex dtypes that can't be natively stored
                group.create_dataset(key, data=str(value))

def convert_mat_to_h5_and_delete(root_directory):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.mat'):
                mat_path = os.path.join(root, file)
                h5_path = os.path.join(root, file.replace('.mat', '.h5'))
                
                try:
                    # Load the MATLAB file
                    mat_data = mat73.loadmat(mat_path)
                    mat_data = mat73.loadmat(mat_path)
                    
                    # Filter metadata
                    data_dict = {k: v for k, v in mat_data.items() if not k.startswith('__')}
                    
                    # Create HDF5 file and save structure
                    with h5py.File(h5_path, 'w') as h5file:
                        save_dict_to_hdf5(h5file, data_dict)
                    
                    print(f"Converted: {mat_path} -> {h5_path}")
                    
                    # Delete original .mat file
                    os.remove(mat_path)
                    print(f"Deleted original: {mat_path}")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    target_dir = "."
    convert_mat_to_h5_and_delete(target_dir)
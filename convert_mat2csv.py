import os
import scipy.io
import pandas as pd

def convert_mat_to_csv_and_delete(root_directory):
    # Traverse through all subdirectories
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            # Check if the file has a .mat extension
            if file.endswith('.mat'):
                mat_path = os.path.join(root, file)
                csv_path = os.path.join(root, file.replace('.mat', '.csv'))
                
                try:
                    # Load the MATLAB file
                    mat_data = scipy.io.loadmat(mat_path)
                    
                    # Filter out metadata (keys starting with '__')
                    data_dict = {k: v for k, v in mat_data.items() if not k.startswith('__')}
                    
                    if not data_dict:
                        print(f"Skipped {file}: No valid data found.")
                        continue

                    # Convert to a DataFrame and save as CSV
                    # Note: This assumes data is in a format pandas can interpret (like arrays)
                    df = pd.DataFrame({k: pd.Series(v.flatten() if hasattr(v, 'flatten') else v) 
                                     for k, v in data_dict.items()})
                    df.to_csv(csv_path, index=False)
                    print(f"Converted: {mat_path} -> {csv_path}")

                    # Delete the original .mat file after successful conversion
                    os.remove(mat_path)
                    print(f"Deleted original: {mat_path}")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    # Replace with the path to your target directory
    target_dir = "."
    convert_mat_to_csv_and_delete(target_dir)
import h5py

def load_h5_to_dict(group):
    """Recursively loads an H5 group back into a dictionary."""
    data = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            data[key] = load_h5_to_dict(item)
        else:
            data[key] = item[()] # [()] extracts the data as a numpy array/scalar
    return data
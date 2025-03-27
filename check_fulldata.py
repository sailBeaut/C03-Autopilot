import h5py

# Open the HDF5 file in read mode
#file_path = "/Users/lennarthubbers/Desktop/processed-20250217_151129.hdf5"  # Replace with your actual file path
file_path = "extended_filtered_data.hdf5"  # Replace with your actual file path

def dat_array(dir):
	with h5py.File(file_path, "r") as f:
		try:
			data = f[str(dir)][()]
			return data
		except:
			print("invalid directory")

def print_struc():
	with h5py.File(file_path, "r") as f:
		def print_structure(name, obj):
			print(name, "->", "Group" if isinstance(obj, h5py.Group) else "Dataset")
		f.visititems(print_structure)

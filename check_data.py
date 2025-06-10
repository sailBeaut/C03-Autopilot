import h5py

#file_path = "/Users/lennarthubbers/Desktop/processed-20250217_151129.hdf5"  # Replace with your actual file path
file_path = "filtered_data_nn_2.hdf5"  # Replace with your actual file path

def dat_array(dir): # reads data from file with given directory
	with h5py.File(file_path, "r") as f: # Open the HDF5 file in read mode
		try:
			data = f[str(dir)][()]
			return data # returns data if proper directory given
		except:
			print("invalid directory")

def print_struc(): # prints structure of the hdf5 file
	with h5py.File(file_path, "r") as f: 
		def print_structure(name, obj):
			print(name, "->", "Group" if isinstance(obj, h5py.Group) else "Dataset")
		f.visititems(print_structure)

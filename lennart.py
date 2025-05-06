import h5py
import numpy as np

def extract_relevant_data(input_file, output_file, runs=12):
	"""Extracts DeltaAil and DeltaElev data from multiple runs and saves to a new HDF5 file."""
	with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dest:
		i =1
		minv = int(5.1517*10**5)
		maxv = minv+7001
		print(minv, maxv)
		for dataset in ['IservoAil', 'IservoElev', 'DeltaDrumElev', 'DeltaDrumAil', 'DynPress']:
			run_path = f'data/aircraft/data/{dataset}'
			if run_path in src:
				data = src[run_path][minv:maxv]  # Extract subset of data
				dest.create_dataset(f'run{i}/{dataset}', data=data)
			else:
				print(f"Warning: {run_path} not found in source file.")
		if run_path in src:
			data = np.array(src[f'data/aircraft/data/DeltaAil'])[minv:maxv]-src[f'data/aircraft/data/DeltaAil'][minv]
			dest.create_dataset(f'run{i}/DeltaAil', data=data)	
		if run_path in src:
			data = np.array(src[f'data/aircraft/data/DeltaElev'])[minv:maxv]-src[f'data/aircraft/data/DeltaElev'][minv]
			dest.create_dataset(f'run{i}/DeltElev', data=data)											
		servo_path = f'data/servo/data/delta_e_t'
		if servo_path in src:
			data = src[servo_path]  # Extract subset of data
			dest.create_dataset(f'run{i}/delta_e_t', data=data)
		else:
			print(f"Warning: {servo_path} not found in source file.")

# Example usage
input_hdf5 = 'new_data.hdf5'  # Replace with your actual input file
output_hdf5 = 'split_new_data.hdf5'  # Replace with your desired output file
extract_relevant_data(input_hdf5, output_hdf5)
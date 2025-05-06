import h5py
import numpy as np

def extract_relevant_data(input_file, output_file, runs=12):
	"""Extracts DeltaAil and DeltaElev data from multiple runs and saves to a new HDF5 file."""
	with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dest:
		for i in [1]:  # Assuming runs are named run1, run2, ..., run12
			for dataset in ['IservoAil', 'IservoElev', 'DeltaDrumElev', 'DeltaDrumAil', 'DynPress']:
				run_path = f'run{i}/aircraft/{dataset}'

				if run_path in src:
					data = src[run_path][4000:11001]  # Extract subset of data
					dest.create_dataset(f'run{i}/aircraft/{dataset}', data=data)
				else:
					print(f"Warning: {run_path} not found in source file.")
			if run_path in src:
				data = np.array(src[f'run{i}/aircraft/DeltaAil'])[4000:11001]-src[f'run{i}/aircraft/DeltaAil'][4000]
				dest.create_dataset(f'run{i}/aircraft/DeltaAil', data=data)	
			if run_path in src:
				data = np.array(src[f'run{i}/aircraft/DeltaElev'])[4000:11001]-src[f'run{i}/aircraft/DeltaElev'][4000]
				dest.create_dataset(f'run{i}/aircraft/DeltElev', data=data)											
			servo_path = f'run{i}/servo/delta_e_t'
			if servo_path in src:
				data = src[servo_path]  # Extract subset of data
				dest.create_dataset(servo_path, data=data)
			else:
				print(f"Warning: {servo_path} not found in source file.")

# Example usage
input_hdf5 = 'new_data.hdf5'  # Replace with your actual input file
output_hdf5 = 'split_new_data.hdf5'  # Replace with your desired output file
extract_relevant_data(input_hdf5, output_hdf5)
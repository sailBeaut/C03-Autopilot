import h5py
import numpy as np
'''
def extract_relevant_data(input_file, output_file, runs=12):
	"""Extracts DeltaAil and DeltaElev data from multiple runs and saves to a new HDF5 file."""
	with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dest:
		for i in [1,3,4,5,6,7,8,9,10,11,12,13]:  # Assuming runs are named run1, run2, ..., run12
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
input_hdf5 = '/Users/lennarthubbers/Desktop/processed-20250217_151129.hdf5'  # Replace with your actual input file
output_hdf5 = 'filtered_data3.hdf5'  # Replace with your desired output file
extract_relevant_data(input_hdf5, output_hdf5)
'''


'''
def extract_relevant_data(input_file, output_file, runs=12):
	"""Extracts DeltaAil and DeltaElev data from multiple runs and saves to a new HDF5 file."""
	with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dest:
		for dataset in ['IservoAil', 'IservoElev', 'DeltaDrumElev', 'DeltaDrumAil', 'DynPress']:
			run_path = f'data/aircraft/data/{dataset}'
			if run_path in src:
				data = src[run_path]  # Extract subset of data
				dest.create_dataset(f'data/aircraft/data/{dataset}', data=data)
			else:
				print(f"Warning: {run_path} not found in source file.")
		if run_path in src:
			data = np.array(src[f'data/aircraft/data/DeltaAil'])
			dest.create_dataset(f'data/aircraft/data/DeltaAil', data=data)	
		if run_path in src:
			data = np.array(src[f'data/aircraft/data/DeltaElev'])
			dest.create_dataset(f'data/aircraft/data/DeltaElev', data=data)											
		servo_path = f'data/servo/data/delta_e_t'
		if servo_path in src:
			data = src[servo_path]  # Extract subset of data
			dest.create_dataset(servo_path, data=data)
		else:
			print(f"Warning: {servo_path} not found in source file.")

# Example usage
input_hdf5 = '/Users/lennarthubbers/Desktop/simlog-20250424_091724.hdf5'  # Replace with your actual input file
output_hdf5 = 'new_data.hdf5'  # Replace with your desired output file
extract_relevant_data(input_hdf5, output_hdf5)'''


import h5py
import numpy as np

def extract_relevant_data(input_file, output_file, run_windows):
    """
    Extracts DeltaAil and DeltaElev data from multiple runs and saves to a new HDF5 file.

    Parameters:
    - input_file (str): Path to the input HDF5 file.
    - output_file (str): Path to the output HDF5 file.
    - run_windows (dict): Dictionary where keys are run numbers and values are (start, end) indices.
    """
    with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dest:
        for run, (start, end) in run_windows.items():
            for dataset in ['IservoAil', 'IservoElev', 'DeltaDrumElev', 'DeltaDrumAil', 'DynPress']:
                run_path = f'run{run}/aircraft/{dataset}'
                if run_path in src:
                    data = src[run_path][start:end]  # Extract subset of data
                    dest.create_dataset(f'run{run}/aircraft/{dataset}', data=data)
                else:
                    print(f"Warning: {run_path} not found in source file.")

            # Process DeltaAil
            delta_ail_path = f'run{run}/aircraft/DeltaAil'
            if delta_ail_path in src:
                data = np.array(src[delta_ail_path][start:end]) - src[delta_ail_path][start]
                dest.create_dataset(f'run{run}/aircraft/DeltaAil', data=data)

            # Process DeltaElev
            delta_elev_path = f'run{run}/aircraft/DeltaElev'
            if delta_elev_path in src:
                data = np.array(src[delta_elev_path][start:end]) - src[delta_elev_path][start]
                dest.create_dataset(f'run{run}/aircraft/DeltaElev', data=data)

            # Process servo data
            servo_path = f'run{run}/servo/delta_e_t'
            if servo_path in src:
                data = src[servo_path][start:end]  # Extract subset of data
                dest.create_dataset(servo_path, data=data)
            else:
                print(f"Warning: {servo_path} not found in source file.")

# Example usage
input_hdf5 = 'extended_filtered_data2.hdf5'  # Replace with your actual input file
output_hdf5 = 'new_filtered_data.hdf5'  # Replace with your desired output file

# Define adjustable windows for each run
run_windows = {
    1: (3500, 16000),
    3: (3500, 19500),
    4: (3500, 20000),
    5: (3500, 19500),
    6: (3500, 19500),
    7: (3500, 19000),
	8: (3500, 17000),
	9: (3500, 19000),
	10: (3500, 16000),
	11: (3000, 19000),
    12: (3000, 16000),
	13: (3000, 17000),
# Add more runs as needed
}

extract_relevant_data(input_hdf5, output_hdf5, run_windows)
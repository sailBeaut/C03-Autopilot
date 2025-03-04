import h5py

def extract_relevant_data(input_file, output_file, runs=12):
    """Extracts DeltaAil and DeltaElev data from multiple runs and saves to a new HDF5 file."""
    with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dest:
        for i in [1,3,4,5,6,7,8,9,10,11,12,13]:  # Assuming runs are named run1, run2, ..., run12
            for dataset in ['DeltaAil', 'DeltaElev', 'IservoAil', 'IservoElev', 'DeltaDrumElev', 'DeltaDrumAil']:
                run_path = f'run{i}/aircraft/{dataset}'
                
                if run_path in src:
                    data = src[run_path][4000:11001]  # Extract subset of data
                    dest.create_dataset(f'run{i}/aircraft/{dataset}', data=data)
                else:
                    print(f"Warning: {run_path} not found in source file.")
            
            servo_path = f'run{i}/servo/delta_e_t'
            if servo_path in src:
                data = src[servo_path][4000:11001]  # Extract subset of data
                dest.create_dataset(servo_path, data=data)
            else:
                print(f"Warning: {servo_path} not found in source file.")

# Example usage
input_hdf5 = '/Users/lennarthubbers/Desktop/processed-20250217_151129.hdf5'  # Replace with your actual input file
output_hdf5 = 'filtered_data.hdf5'  # Replace with your desired output file
extract_relevant_data(input_hdf5, output_hdf5)
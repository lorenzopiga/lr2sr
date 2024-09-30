import numpy as np
import os
import multiprocessing
import torch

# Function to convert displacement field to positions for a given chunk
def dis2pos(dis_field, boxsize, Ng_chunk):
    """Convert displacement field to positions for a given chunk."""
    cellsize = boxsize / Ng_chunk
    
    # Generate lattice for this chunk
    lattice = np.arange(Ng_chunk) * cellsize + 0.5 * cellsize

    pos = dis_field.copy()

    # Add lattice positions to displacements (for each axis)
    pos[2] += lattice
    pos[1] += lattice.reshape(-1, 1)
    pos[0] += lattice.reshape(-1, 1, 1)

    # Ensure positions are within the box bounds
    pos[pos < 0] += boxsize
    pos[pos > boxsize] -= boxsize

    return pos

# Function to process a single SR displacement chunk and save the positions
def process_chunk(chunk_file, chunk_files_dir, boxsize, Ng_chunk, output_dir, gpu_id):
    """Process a single SR displacement chunk and save the positions."""
    
    # Set the GPU device for this process
    torch.cuda.set_device(gpu_id)
    print(f"Processing {chunk_file} on GPU {gpu_id}")

    # Load the displacement chunk from file
    chunk_path = os.path.join(chunk_files_dir, chunk_file)
    chunk_disp = np.load(chunk_path)  # Load the displacement chunk

    # Apply dis2pos to convert displacements to positions for this chunk
    positions = dis2pos(chunk_disp, boxsize, Ng_chunk)

    # Reshape positions from (3, Ng_chunk, Ng_chunk, Ng_chunk) to (Ng_chunk^3, 3)
    positions_reshaped = positions.reshape(3, Ng_chunk**3).transpose()

    # Save this chunk's positions to the output directory
    output_file = os.path.join(output_dir, f"sr_positions_{chunk_file.split('_')[-1]}")
    np.save(output_file, positions_reshaped)
    
    print(f"Processed and saved positions for chunk: {chunk_file} on GPU {gpu_id}")

# Main script entry point
if __name__ == "__main__":
    # Define parameters
    chunk_files_dir = 'GPU_2nd_test'  # Directory containing displacement chunks
    boxsize = 100000  # Box size in kpc/h (set as required for your simulation)
    Ng_chunk = 128  # Number of particles per axis in each chunk (128 in this case)
    output_dir = '/path/to/output_positions'  # Directory where positions will be saved
    gpu_list = [0, 1, 2, 3]  # List of available GPUs to use
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all chunk files in the directory (assuming they start with 'chunk_disp_')
    chunk_files = [f for f in os.listdir(chunk_files_dir) if f.startswith("chunk_disp_")]

    # Use multiprocessing to process chunks in parallel, distributing across GPUs
    with multiprocessing.Pool(processes=len(gpu_list)) as pool:
        pool.starmap(
            process_chunk,
            [(chunk_file, chunk_files_dir, boxsize, Ng_chunk, output_dir, gpu_list[i % len(gpu_list)])
             for i, chunk_file in enumerate(chunk_files)]
        )

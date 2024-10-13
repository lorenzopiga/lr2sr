import numpy as np
import torch
from map2map import models
from map2map.norms import cosmology
import argparse
import os
import time
import multiprocessing

# Global variables to hold the model and device across processes
model = None
device = None
device_list = [0, 1, 2, 3, 4, 5]#, 6], 7]  # List of GPU devices

def initialize_worker(model_path, device_id):
    global model, device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = torch.device('cuda')

    # Load the model and move it to the appropriate device
    upsample_fac = 8
    in_channels = out_channels = 6
    model = models.G(in_channels, out_channels, upsample_fac)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval().to(device)
    del state

def narrow_like(sr_box, tgt_Ng):
    """ sr_box in shape (Nc, Ng, Ng, Ng), trim to (Nc, tgt_Ng, tgt_Ng, tgt_Ng), better to be even """
    width = np.shape(sr_box)[1] - tgt_Ng
    half_width = width // 2
    begin, stop = half_width, tgt_Ng + half_width
    return sr_box[:, begin:stop, begin:stop, begin:stop]

def cropfield(field, idx, reps, crop, pad):
    """input field in shape of (Nc, Ng, Ng, Ng),
    crop idx^th subbox in reps grid with padding"""
    start = np.array(np.unravel_index(idx, reps)) * np.array(crop)  # Element-wise multiplication
    x = field.copy()
    for d, (i, N, (p0, p1)) in enumerate(zip(start, crop, pad)):
        x = x.take(range(i - p0, i + N + p1), axis=1 + d, mode='wrap')
    return x

def sr_field(lr_field, tgt_size, device, model):
    """input *normalized* lr_field in shape of (Nc, Ng, Ng, Ng),
    return unnormalized sr_field trimmed to tgt_size^3
    """
    lr_field = np.expand_dims(lr_field, axis=0)
    lr_field = torch.from_numpy(lr_field).float()
    lr_field = lr_field.to(device)

    with torch.no_grad():
        sr_box = model(lr_field)
    sr_box = sr_box.cpu().numpy()
    sr_disp = cosmology.disnorm(sr_box[0, 0:3,], z=redshift, undo=True)
    sr_disp = narrow_like(sr_disp, tgt_size)
    sr_vel = cosmology.velnorm(sr_box[0, 3:6,], z=redshift, undo=True)
    sr_vel = narrow_like(sr_vel, tgt_size)
    return sr_disp, sr_vel

# dis2pos function to convert displacements to positions
def dis2pos(dis_field, boxsize, Ng):
    """Assume 'dis_field' is in order of `pid` that aligns with the Lagrangian lattice,
    and dis_field.shape = (3, Ng, Ng, Ng)
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize

    pos = dis_field.copy()
    pos[2] += lattice
    pos[1] += lattice.reshape(-1, 1)
    pos[0] += lattice.reshape(-1, 1, 1)

    pos[pos < 0] += boxsize
    pos[pos > boxsize] -= boxsize

    return pos

def process_chunk(lr_box, idx, reps, crop, pad, tgt_size, redshift, chunk_files, Lbox_kpc, Ng_sr, random_sample=True):
    global model, device
    start_time = time.time()

    chunk = cropfield(lr_box, idx, reps, crop, pad)
    chunk_disp, chunk_vel = sr_field(chunk, tgt_size, device, model)

    # If random sampling is enabled, sample 5% of the particles
    if random_sample:
        total_particles = Ng_sr**3
        sample_size = int(0.05 * total_particles)  # 5% of the particles
        sampled_indices = np.random.choice(total_particles, sample_size, replace=False)
        chunk_disp = chunk_disp.reshape(3, total_particles).transpose()[sampled_indices].transpose()

    # Apply dis2pos to the displacement chunk to get the positions
    sr_pos_chunk = dis2pos(chunk_disp, Lbox_kpc, tgt_size)

    # Reshape the positions to save them in (Ng_sr^3, 3) format, or sampled version if random_sample=True
    sr_pos_chunk = sr_pos_chunk.reshape(3, -1).transpose()

    # Save the SR position chunk
    filename = os.path.join(chunk_files, f"chunk_pos_{idx}.npy")
    np.save(filename, sr_pos_chunk)

    elapsed_time = time.time() - start_time
    print(f"Iteration {idx+1}/{tot_reps} done in {elapsed_time:.4f} seconds", flush=True)
    return idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lr2sr')
    parser.add_argument('--model-path', required=True, type=str, help='path of the generative model')
    parser.add_argument('--redshift', required=True, type=float, help='redshift of the model')
    parser.add_argument('--lr-input', required=True, type=str, help='path of the lr input')
    parser.add_argument('--sr-path', required=True, type=str, help='path to save sr output')
    parser.add_argument('--Lbox-kpc', default=100000, type=float, help='LR/HR/SR Boxsize, in kpc/h')
    parser.add_argument('--nsplit', default=4, type=int, help='split the LR box into chunks to apply SR')
    parser.add_argument('--chunk_files', default=4, type=str, help='outpath for the chunk_files')
    parser.add_argument('--random-sample', action='store_true', help='Enable 5% random sampling of particles')

    args = parser.parse_args()
    model_path = args.model_path
    redshift = args.redshift

    n_split = args.nsplit
    lr_box = np.load(args.lr_input, mmap_mode='r')  # Use memory mapping for large data
    size = lr_box.shape[1:]  # Assuming lr_box has shape (Nc, Ng, Ng, Ng)

    # Calculate chunk size
    chunk_size = tuple(s // n_split for s in size)

    # Calculate reps and padding
    reps = tuple(s // cs for s, cs in zip(size, chunk_size))
    tot_reps = int(np.prod(reps))

    pad = 3
    pad = np.broadcast_to(pad, (len(size), 2))

    # Calculate target size
    upsample_fac = 8
    tgt_size = chunk_size[0] * upsample_fac
    tgt_chunk = np.broadcast_to(tgt_size, size)

    Ng_sr = size[0] * upsample_fac
    chunk_files = args.chunk_files

    existing_chunks = {int(f.split('_')[-1].split('.')[0]) for f in os.listdir(chunk_files) if f.startswith("chunk_pos_")}
    print(f"Existing chunks: {sorted(existing_chunks)}")

    tasks = [(lr_box, idx, reps, chunk_size, pad, tgt_size, redshift, chunk_files, args.Lbox_kpc, Ng_sr, args.random_sample)
             for idx in range(tot_reps) if idx not in existing_chunks]

    # Sequential processing with smaller chunks (by default)
    chunksize = 1  # Each GPU gets one chunk at a time in sequential order

    # Use multiprocessing with an initializer to load the model and set up CUDA once per worker
    with multiprocessing.Pool(processes=len(device_list), initializer=initialize_worker, initargs=(model_path, device_list[0])) as pool:
        pool.starmap(process_chunk, tasks, chunksize=chunksize)


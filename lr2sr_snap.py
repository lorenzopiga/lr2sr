import psutil

# Ottieni le informazioni sulla memoria
memory_info = psutil.virtual_memory()

# Stampa la RAM totale e disponibile
print(f"RAM Totale: {memory_info.total / (1024 ** 3):.2f} GB")
print(f"RAM Disponibile: {memory_info.available / (1024 ** 3):.2f} GB")
print(f"RAM Usata: {memory_info.used / (1024 ** 3):.2f} GB")
print(f"Percentuale di utilizzo della RAM: {memory_info.percent}%")


import numpy as np
import torch
from map2map import models
from map2map.norms import cosmology

from bigfile import BigFile
import argparse
import os
import sys



# Load parameter and model
#--------------------------
parser = argparse.ArgumentParser(description='lr2sr')
parser.add_argument('--model-path',required=True,type=str,help='path of the generative model')
parser.add_argument('--redshift',required=True,type=float,help='redshift of the model')
parser.add_argument('--lr-input',required=True,type=str,help='path of the lr input')
parser.add_argument('--sr-path',required=True,type=str,help='path to save sr output')
parser.add_argument('--Lbox-kpc',default=100000,type=float,help='LR/HR/SR Boxsize, in kpc/h')
parser.add_argument('--nsplit',default=4,type=int,help='split the LR box into chunks to apply SR')
parser.add_argument('--chunk_files',default=4,type=str,help='outpath for the chunk_files')

args = parser.parse_args()
model_path = args.model_path
redshift = args.redshift

# load model
upsample_fac = 8
in_channels = out_channels = 6
model = models.G(in_channels,out_channels,upsample_fac)

#device = torch.device('cuda')
#torch.cuda.device(device)
device = torch.device('cpu')

state = torch.load(model_path, map_location=device)

model.load_state_dict(state['model'])
print('load model state at epoch {}'.format(state['epoch']))
epoch = state['epoch']
del state

model.eval()
model.to(device)

#--------------------------
def narrow_like(sr_box,tgt_Ng):
    """ sr_box in shape (Nc,Ng,Ng,Ng),trim to (Nc,tgt_Ng,tgt_Ng,tgt_Ng), better to be even """
    width = np.shape(sr_box)[1] - tgt_Ng
    half_width = width // 2
    begin,stop = half_width,tgt_Ng + half_width
    return sr_box[:,begin:stop,begin:stop,begin:stop]
    
def cropfield(field,idx,reps,crop,pad):
    """input field in shape of (Nc,Ng,Ng,Ng),
    crop idx^th subbox in reps grid with padding"""
    start = np.unravel_index(idx, reps) * crop  # find coordinate of idx in reps grid
    x = field.copy()
    for d, (i, N, (p0, p1)) in enumerate(zip(start, crop, pad)):
        x = x.take(range(i - p0, i + N + p1), axis=1 + d, mode='wrap')
    return x

def sr_field(lr_field,tgt_size):
    """input *normalized* lr_field in shape of (Nc,Ng,Ng,Ng),
    return unnormalized sr_field trimmed to tgt_size^3
    """
    lr_field = np.expand_dims(lr_field, axis=0)
    print("lr_field in sr_field is: ", np.shape(lr_field))
    lr_field = torch.from_numpy(lr_field).float()
    print("lr_field in sr_field is: ", np.shape(lr_field))
    lr_field = lr_field.to(device)
    print("I'm about to call torch")

    with torch.no_grad():
        sr_box = model(lr_field)
    print("I've called torch")
    sr_box = sr_box.cpu().numpy()
    print("sr_box in sr_field is: ", np.shape(sr_box))
    sr_disp = cosmology.disnorm(sr_box[0,0:3,],z=redshift,undo=True)
    sr_disp = narrow_like(sr_disp,tgt_size)
    print("sr_disp in sr_field is: ", np.shape(sr_disp))
    sr_vel = cosmology.velnorm(sr_box[0,3:6,],z=redshift,undo=True)
    sr_vel = narrow_like(sr_vel,tgt_size)
    print("sr_vel in sr_field is: ", np.shape(sr_vel))
    return sr_disp,sr_vel
    
################################ crop the input box, sr operation, and piece back ###########################

n_split = args.nsplit # split the lr input into n_split^3 chunks
lr_box = np.load(args.lr_input) # lr_input in shape of (Nc,Ng,Ng,Ng)
print("lr input is: ", np.shape(lr_box))
size = lr_box.shape[1:]
size = np.asarray(size)
ndim = len(size)

chunk_size = size // n_split
crop = np.broadcast_to(chunk_size, size.shape)

reps = size // crop
tot_reps = int(np.prod(reps))
print("tot_reps is: ", np.shape(tot_reps))
pad = 3
pad = np.broadcast_to(pad, (ndim, 2))
print("pad is: ", np.shape(pad))

tgt_size = crop[0]*upsample_fac
tgt_chunk = np.broadcast_to(tgt_size, size.shape) 
print("tgt_chunk is: ", np.shape(tgt_chunk))

Ng_sr = size[0]*upsample_fac 
print("Ng_sr: ", np.shape(Ng_sr))
#disp_field = np.zeros([3,Ng_sr,Ng_sr,Ng_sr])
#print("disp_field is: ", np.shape(disp_field))
#vel_field = np.zeros([3,Ng_sr,Ng_sr,Ng_sr])
#print("vel_field is: ", np.shape(vel_field))

chunk_files = args.chunk_files
for idx in range(0,tot_reps):
    print("I'm in the for loop!")
    chunk = cropfield(lr_box,idx,reps,crop,pad)
    print('chunk is: ', np.shape(chunk))
    chunk_disp,chunk_vel = sr_field(chunk,tgt_size)
    print('chunk disp: ', np.shape(chunk_disp))
    #chunk_files = args.chunk_files
    #np.save(outpath+"{_}".format(idx+1),chunk_disp)
    filename = os.path.join(chunk_files, f"chunk_disp_{idx}.npy")
    np.save(filename, chunk_disp)
    #print("output saved!")
    ns = np.unravel_index(idx, reps) * tgt_chunk  # new start point
    print('ns: ', ns)
    #disp_field[:,ns[0]:ns[0]+tgt_size,ns[1]:ns[1]+tgt_size,ns[2]:ns[2]+tgt_size] = chunk_disp
    #print('disp_field is: ', disp_field)
    #vel_field[:,ns[0]:ns[0]+tgt_size,ns[1]:ns[1]+tgt_size,ns[2]:ns[2]+tgt_size] = chunk_vel
    #print("{}/{} done".format(idx+1,tot_reps),flush=True)

#disp_field = np.float32(disp_field)
#vel_field = np.float32(vel_field)

################################ transfer back to position column ###########################
def dis2pos(dis_field,boxsize,Ng):
    """Assume 'dis_field' is in order of `pid` that aligns with the Lagrangian lattice,
    and dis_field.shape = (3,Ng,Ng,Ng)
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize

    pos = dis_field.copy()

    pos[2] += lattice
    pos[1] += lattice.reshape(-1, 1)
    pos[0] += lattice.reshape(-1, 1, 1)

    pos[pos<0] += boxsize
    pos[pos>boxsize] -= boxsize

    return pos

#Lbox = args.Lbox_kpc
#sr_pos = dis2pos(disp_field,Lbox,Ng_sr)
#sr_pos = sr_pos.reshape(3,Ng_sr*Ng_sr*Ng_sr).transpose()
#vel_field = vel_field.reshape(3,Ng_sr*Ng_sr*Ng_sr).transpose()

# output the position and velocity block
#path = args.sr_path
#os.makedirs(path, exist_ok=True)

#dest=BigFile(path,create=1)

#blockname='Position'
#dest.create_from_array(blockname,sr_pos)

#blockname='Velocity'
#dest.create_from_array(blockname,vel_field)

#print ("Generated SR column in ",path)


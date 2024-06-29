"""
This script preprocess the snapshot of N-body simulation into 3D image with 6 channels,
output is in shape of (Nc,Ng,Ng,Ng), giving the normalized {displacement + velocity} field arranged by the original grid of the tracer particles
"""

import numpy as np
from bigfile import File
import argparse
import os, sys
from map2map.norms import cosmology
import readsnap_mod as rs
import numpy as np

def pos2dis(pos, boxsize, Ng):
    """Assume `pos` is ordered in `pid` that aligns with the Lagrangian lattice,
    and all displacement must not exceed half box size.
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize

    pos[..., 0] -= lattice.reshape(-1, 1, 1)
    pos[..., 1] -= lattice.reshape(-1, 1)
    pos[..., 2] -= lattice

    pos -= np.rint(pos / boxsize) * boxsize

    return pos


def get_nonlin_fields(inpath, outpath):
    """
    inpath is LR simulation snapshot in bigfile format (from MP-Gadget)
    outpath is numpy array in shape (Nc,Ng,Ng,Ng)
    """
    
    N_files = 64
    for i in range(N_files):
        dirin=inpath+'snap_062.{}'.format(i)
        #format(snap. snap, i)
        print('Read snap ', i)
        print(dirin)
        header = rs.snapshot_header(dirin) # reads snapshot header
        
        if (i==0):
            pos = rs.read_block(dirin, "POS ", parttype=1, verbose= True)
            pos=pos/1000.0
            vel = rs.read_block(dirin, "VEL ", parttype=1, verbose= True)
            #Ng_ith = header.npart**(1/3)
            #Ng.append(Ng_ith[1])

            #print('Ng is: ', Ng)

        else:
            pos1 = rs.read_block(dirin, "POS ", parttype=1, verbose= True)
            pos1=pos1/1000.0
            pos = np.append(pos, pos1, axis=0)
 
            vel1 = rs.read_block(dirin, "VEL ", parttype=1, verbose= True)
            vel= np.append(vel, vel1, axis= 0)
            
            

    print(len(pos))
    
    print('Size array Position tot: ', pos.shape)
    print('Size array Velocity tot: ', vel.shape)


    boxsize = 1000000
    redshift = 1./header.time - 1
    
    Ng = header.npart**(1/3)
    print('Ng is: ', Ng)
    Ng =int(np.floor( Ng[1]))
    Ng = 1024
    print('Ng is: ', Ng)
    
    pos = pos.reshape(Ng, Ng, Ng, 3)
    
    vel = vel.reshape(Ng, Ng, Ng, 3)

    dis = pos2dis(pos, boxsize, Ng)
    del pos
    print(dis)

    dis = dis.astype('f4')
    vel = vel.astype('f4')
    
    dis = np.moveaxis(dis,-1,0)
    vel = np.moveaxis(vel,-1,0)
    
    disp = cosmology.disnorm(dis,z=redshift)
    velocity = cosmology.velnorm(vel,z=redshift)
    catnorm = np.concatenate([disp,velocity],axis=0)
    catnorm = catnorm.astype('f4')
    print ("z=%.1f"%redshift,"catnorm shape:",np.shape(catnorm))
    
    np.save(outpath,catnorm)
    
#-------------------------------------------------------------------    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--inpath',required=True,type=str,help='path of the LR input snapshot')
    parser.add_argument('--outpath',required=True,type=str,help='path of the output')
    
    args = parser.parse_args()
    
    get_nonlin_fields(args.inpath, args.outpath)
    
    
    
    
    
    

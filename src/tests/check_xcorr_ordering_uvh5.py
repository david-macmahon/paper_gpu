import numpy as np
from pyuvdata import UVData
import argparse
import h5py
import json

NCHANS = 1536
INTSPEC = 131072*2

def signed_int(x):
    """Return two's complement interpretation 
       for 4bit integers"""
    if (x&0x8):
        return (x - 16)
    else:
        return x

def gen_tvg_pol(pol, mode='ramp'):    
    if mode=='ramp':
        ramp = np.arange(2**13, dtype='>B')
        tv = ramp + pol
    if mode=='const':
        tv = np.zeros(NCHANS*4, dtype='>B')
        tv = tv + pol
    tv_imag = np.asarray([signed_int(x) for x in (tv&(0x0f))])
    tv_real = np.asarray([signed_int(x) for x in (tv>>4)])
    tv = tv_real + 1j*tv_imag

    return tv

parser = argparse.ArgumentParser(description='Sanity check on uvh5 files in tvg mode')
parser.add_argument('uvh5_file', type=str, default=None,
                    help = 'File recorded by hera_catcher')
parser.add_argument('-d','--check_data',action='store_true',default=False,
                    help='Check the content of the files')
args=parser.parse_args()

fp = h5py.File(args.uvh5_file)
cminfo = json.loads(fp['Header']['extra_keywords']['cminfo'].value)
fp.close()

uvd = UVData()
uvd.read_uvh5(args.uvh5_file, run_check=False)
ant_pairs = uvd.get_antpairs()

shape = list(np.shape(uvd.get_data(ant_pairs[0])))
zeros_all = np.zeros(shape)
zeros_pol = np.zeros(shape[0:-1])
pols = ['xx','yy','xy','yx']
snap_pol_map = {'e2':0,  'n0':1,
                'e6':2,  'n4':2,
                'e10':4, 'n8':5}

ants = []
for (a1,a2) in ant_pairs:
    try:
        if not np.all(uvd.get_data(a1,a2) == zeros_all):
            ants.append((a1,a2))
            #for i in range(4):
            #    if np.all(np.imag(uvd.get_data(a1,a2)[:,:,i]) == zeros_pol):
            #        print pols[i], a1, a2
    except(KeyError):
        pass

print('Unique antennas', list(set([a[0] for a in ants])))

if args.check_data:
    for a1,a2 in ants:
        if (a1<350)&(a2<350):
            idx_a1 = cminfo['antenna_numbers'].index(a1)
            idx_a2 = cminfo['antenna_numbers'].index(a2)
            snaploc_a1 = snap_pol_map[cminfo['correlator_inputs'][idx_a1][0][:-11]]
            snaploc_a2 = snap_pol_map[cminfo['correlator_inputs'][idx_a2][0][:-11]]

            tspec = gen_tvg_pol(snaploc_a1)*np.conj(gen_tvg_pol(snaploc_a2))
            tspec = np.sum(tspec.reshape(-1,4),axis=1)[:NCHANS]

            print(a1,a2, np.all(np.equal(tspec, uvd.get_data(a2,a1)[0,:,0]/INTSPEC)))

            tspec = gen_tvg_pol(snaploc_a1+1)*np.conj(gen_tvg_pol(snaploc_a2+1))
            tspec = np.sum(tspec.reshape(-1,4),axis=1)[:NCHANS]

            print(a1,a2, np.all(np.equal(tspec, uvd.get_data(a2,a1)[0,:,1]/INTSPEC)))

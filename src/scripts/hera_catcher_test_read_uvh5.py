import numpy as np
#import matplotlib.pyplot as plt
import h5py
import redis
import argparse

parser = argparse.ArgumentParser(description='Read uvh5 files written out by the BDA catcher hashpipe pipeline and plot data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('fname', type=str, help='uvh5 file to read')
args = parser.parse_args()

fp = h5py.File(args.fname,'r')
data = fp['Data']['visdata'].value

# Check time array
bdaconfig = '/tmp/config.txt'

baselines = {}
for n in range(4):
    baselines[n] = []

bdaconfig = np.loadtxt(configfile, dtype=np.int)
for i,t in enumerate(bdaconfig[:,2]):
    if (t==0): continue
    n = int(np.log(t)/np.log(2))
    if (n==4): n = 3
    baselines[n].append((bdaconfig[i,0], bdaconfig[i,1]))



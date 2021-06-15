#! /usr/bin/env python

import sys
import numpy as np

NTIMES = 2048
NTIME_FAST = 4
NCHAN = 384
NANT  = 192
NPOL  = 2
NCPLX = 2


with open(sys.argv[1], 'r') as fh:
    raw = fh.read()

x = np.fromstring(raw, dtype='b')

x = x.reshape(NTIMES/NTIME_FAST, NCHAN, NANT, NPOL, NCPLX, NTIME_FAST)

tvg = np.zeros([NANT, NPOL, NCHAN, NCPLX])

# Generate a test vector valid for all times
for a in range(NANT):
    for p in range(NPOL):
        chan_ramp = (np.arange(NCHAN) + 2*(a%3) + p) % 256
        chan_ramp_r = chan_ramp >> 4
        chan_ramp_i = chan_ramp & 0xf
        chan_ramp_r[chan_ramp_r > 7] -= 16
        chan_ramp_i[chan_ramp_i > 7] -= 16
        tvg[a,p,:,1] = chan_ramp_r
        tvg[a,p,:,0] = chan_ramp_i

#print x[0,:,0,0,0,0]
#print tvg[0,0,:,0]
#exit()

for a in range(12):
    ok = True
    for p in range(NPOL):
        for t in range(NTIMES/NTIME_FAST):
            for t_fast in range(NTIME_FAST):
                for cplx in range(NCPLX):
                    ok = ok and np.all(tvg[a, p, :, cplx] == x[t, :, a, p, cplx, t_fast])
    print("Antenna %d: OK?:" % a, ok)
    #print tvg[a,0,:,:], x[0,:,a,0,:,0]

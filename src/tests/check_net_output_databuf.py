#! /usr/bin/env python

import sys
import numpy as np

NTIMES = 2048
NTIME_PER_PKT = 2
NCHAN_PER_PKT = 384
NANT_PER_PKT = 3
NANT_TOTAL = 192
NPOL_PER_PKT = 2


with open(sys.argv[1], 'r') as fh:
    raw = fh.read()

x = np.fromstring(raw, dtype='B')

x = x.reshape(NTIMES/NTIME_PER_PKT, NANT_TOTAL, NCHAN_PER_PKT, NTIME_PER_PKT, NPOL_PER_PKT)

tvg = np.zeros([NTIMES/NTIME_PER_PKT, NANT_TOTAL, NCHAN_PER_PKT, NTIME_PER_PKT, NPOL_PER_PKT])

ramp = np.arange(384) % 256
for t in range(NTIMES/NTIME_PER_PKT):
    for t_pkt in range(NTIME_PER_PKT):
        for a in range(NANT_TOTAL):
            for p in range(NPOL_PER_PKT):
                tvg[t, a, :, t_pkt, p] = (np.arange(384) + 2*(a%3) + p) % 256

for a in range(NANT_TOTAL):
    print("Antenna %d: OK?:" % a, end=' ')
    print(np.all(tvg[:,a,:,:,:] == x[:,a,:,:,:]))

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

x = np.fromstring(raw, dtype=np.int32)

print("All values equal to 0?:", np.all(x == 0))
print("All values divisible by 2048?:", np.all((x % 2048) == 0))
print("All values divisible by 262144?:", np.all((x % 262144) == 0))

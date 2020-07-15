import numpy as np

NANTS = 192
NCHANS = 8192 / 4 * 3 / 4
NXENG = 16
NWORDS = 2 * NANTS * (NANTS+1) / 2 * 4 * NCHANS
NBL = (NANTS * (NANTS+1)) / 2

def get_bl_order(n_ants):
    """
    Return the order of baseline data output by a CASPER correlator
    X engine.
    
    Extracted from the corr package -- https://github.com/ska-sa/corr
    """
    order1, order2 = [], []
    for i in range(n_ants):
        for j in range(int(n_ants/2),-1,-1):
            k = (i-j) % n_ants
            if i >= k: order1.append((k, i))
            else: order2.append((i, k))
    order2 = [o for o in order2 if o not in order1]
    return tuple([o for o in order1 + order2])

with open("/tmp/packet.bin", "r") as fh:
    x = np.fromstring(fh.read(), dtype='>i')

bl_order = get_bl_order(NANTS)
x = x.reshape(2, NCHANS, NBL, 4, 2)

print(np.all(x==0))

for bn, bl in enumerate(bl_order):
    r = x[0,:,bn,0,0]
    i = x[0,:,bn,0,1]
    if not (np.all(r==0) and np.all(i==0)):
        print(bl[0], bl[1], r, i)

import time
import socket
import struct
import numpy

NANTS = 192
NCHANS = 8192 / 4 * 3
BYTES_PER_PACKET = 4096
MCNT_STEP = 2**18
NX = 8
TIME_DEMUX = 2
NT = 2
PERIOD = 10

DESTIP = socket.gethostbyname("catcher")
DESTPORT = 10000

n_bls = (NANTS * (NANTS+1)) / 2 * 4
n_bytes = TIME_DEMUX * NCHANS * n_bls * 2 * 4 # real/imag, bytes-per-word
n_pkts = (n_bytes / BYTES_PER_PACKET)
n_pkts_per_x = n_pkts / NX

print("MBytes per dump: %.2f" % (n_bytes / 1e6))
print("Packets per dump: %.4f" % (n_bytes / float(BYTES_PER_PACKET)))
print("Packets per Xeng: %.4f" % n_pkts_per_x)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mcnt = 1310720
data = struct.pack('>2i', 1, 2) * (BYTES_PER_PACKET/8)
assert len(data) == BYTES_PER_PACKET

while(True):
    start = time.time()
    offset = 0
    for pkt in range(n_pkts_per_x/TIME_DEMUX):
        #if pkt % 100000 == 0:
        #    print pkt
        for t in range(TIME_DEMUX):
            for xeng in range(NX):
                p = struct.pack('>QLHH', mcnt+NT*t, offset, xeng, BYTES_PER_PACKET) + data
                s.sendto(p, (DESTIP, DESTPORT))
            time.sleep(1e-6)
        offset += BYTES_PER_PACKET
    elapsed = time.time() - start
    mcnt += MCNT_STEP
    wait_time = PERIOD - elapsed
    if wait_time > 0:
        time.sleep(wait_time)
    total_time = time.time() - start
    print("dump sent (MCNT: %d) in %.2fs at %.2f Gb/s" % (mcnt, total_time, n_bytes*8 / elapsed / 1e9))


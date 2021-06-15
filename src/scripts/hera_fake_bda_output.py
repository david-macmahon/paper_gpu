#!/usr/bin/env python

# Imitate 16 xengs, 2 times, all baselines 
# Send packets to catcher imitating the output
# of all the xengs in the chain.

import numpy as np
import argparse
import struct
import socket
import time
import redis

parser = argparse.ArgumentParser(description='Generate FAKE output to test catcher pipeline with baseline dependent averaging',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default=None, 
                    help='BDA config file')
parser.add_argument('--catcher', type=str, default = '10.80.40.251',
                    help='IP address of the Catcher machine')
args = parser.parse_args()

if args.config:
   config = args.config
else:
   r = redis.Redis('redishost')
   config = r.get('bda:config')

Na = 352   # antennas
Nx = 16    # xeng
Nc = 384   # chan
Nt = 2     # demux
Ns = 4     # stokes
Nbins = 4  # number of diff integration bins
udp_ip = args.catcher #'10.80.40.251'
udp_port = 10000
inttimes = np.asarray([1,2,4,8,16])

int_bin = {}
int_bin['baselines'] = {}
int_bin['data']      = {}

fakereal = 1
fakeimag = 2

for n in range(Nbins):
    int_bin['baselines'][n] = []

    # Constant
    #int_bin['data'][n] = (2**n)*np.ones(1024, dtype=np.int32)
    #int_bin['data'][n][1::2] = -2*(2**n)

    # Ramp
    int_bin['data'][n] = (2**n)*np.repeat((np.arange(128, 
                         dtype=np.int32)+fakereal),8)
    int_bin['data'][n][1::2] = -1*(2**n) * np.repeat((np.arange(128, 
                                dtype=np.int32)+fakeimag),4)

    # convert to binary data for speedup
    int_bin['data'][n] = np.array(int_bin['data'][n]).byteswap().tostring()


# Populate baseline pairs from config file
bdaconfig = np.loadtxt(config, dtype=np.int)
for i,t in enumerate(bdaconfig[:,2]):
    if (t==0): continue
    n = int(np.log(t)/np.log(2))
    if (n==4): n = 3
    int_bin['baselines'][n].append((bdaconfig[i,0], bdaconfig[i,1]))

for b in list(int_bin['baselines'].keys()):
    print(b,len(int_bin['baselines'][b]))

bcnt = 0; mcnt = 0; ctr = 0;
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sleep_time = 10e-7

while True:
   ctr += 2
   mcnt = int(500e6 * ctr / (2 * 8192))
   for ns in np.logspace(1, Nbins, num=Nbins, base=2, dtype=np.int):
       if (ctr%ns == 0):
           nb = int(np.log2(ns)) - 1
           print('Sending: %d \tBaselines: %d \tBcnt: %d' % (ns, len(int_bin['baselines'][nb]), bcnt))
           for (a0,a1) in int_bin['baselines'][nb]:
               for xeng_id in range(Nx*Nt):
                   b = mcnt+((xeng_id//Nx)*2)
                   pkt = struct.pack('>1Q2I4H', b, bcnt, 0, a0, a1, xeng_id, 4096) + int_bin['data'][nb]
                   sock.sendto(pkt, (udp_ip, udp_port))
                   time.sleep(sleep_time)
                   pkt = struct.pack('>1Q2I4H', b, bcnt, 1, a0, a1, xeng_id, 4096) + int_bin['data'][nb]
                   sock.sendto(pkt, (udp_ip, udp_port))
                   time.sleep(sleep_time)
                   pkt = struct.pack('>1Q2I4H', b, bcnt, 2, a0, a1, xeng_id, 4096) + int_bin['data'][nb]
                   sock.sendto(pkt, (udp_ip, udp_port))
                   time.sleep(sleep_time)
               bcnt += 1


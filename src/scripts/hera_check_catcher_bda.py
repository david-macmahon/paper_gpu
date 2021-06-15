# BDA
# Check packet structure and contents in test mode

import numpy as np
import struct
import socket
import argparse 

N_ANTS_DATA = 192       # antennas
N_bl_per_block = 256    # baselines within each block
N_MAX_INTTIME = 8
N_BDABUF_BINS = 4
N_CHAN_TOTAL = 6144
N_STOKES = 4
NCHANS = 1536
N_CHAN_CATCHER = 1536
INTSPEC = 131072

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

def decode_packet(packet):
   # mcnt, bcnt, offset, ant0, ant1, xeng_id, payload_len
   p = struct.unpack('>1Q2I4H1024i', packet)
   time, bcnt, offset, ant0, ant1, xeng_id, payload_len = p[0:7]
   data = np.asarray(p[7:])
   return time, bcnt, offset, ant0, ant1, xeng_id, payload_len, data


parser = argparse.ArgumentParser(description='Test packet format and contents for BDA',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--host', type=str, default="10.10.10.222",
                     help='Host to capture data')
parser.add_argument('-p', dest='port', type=int, default=10000,
                     help='Host port to receive data')
parser.add_argument('-n', dest='n_inputs', type=int, default=704,
                     help='Nant*Npol that xGPU has been configured')
parser.add_argument('-v', dest='verbose', action='store_true', default=False,
                     help='Print some contents of each packet')
parser.add_argument('-c', dest='check', action='store_true', default=False,
                     help='Check the packet contents')
parser.add_argument('--config', dest='bdaconfig', type=str, default='/tmp/bdaconfig.txt',
                     help='BDA config file')
parser.add_argument('--snap', action='store_true', default=False,
                     help='Set if the test vectors are coming from SNAPs')

args = parser.parse_args()

snap_pol_map = {'e2':0,  'n0':1,
                'e6':2,  'n4':2,
                'e10':4, 'n8':5}

int_bin = {}
int_bin['baselines'] = {}
int_bin['data']      = {}

fakereal = 1
fakeimag = 2

for n in range(N_BDABUF_BINS):
    int_bin['baselines'][n] = []

    # Ramp
    int_bin['data'][n] = (2**n)*np.repeat((np.arange(384)+fakereal),8)
    int_bin['data'][n][1::2] = -1*(2**n) * np.repeat((np.arange(384)+fakeimag),4)

    # Const
    #int_bin['data'][n] = (2**n)*np.ones(1024)
    #int_bin['data'][n][1::2] = -2*(2**n)

conf = np.loadtxt(args.bdaconfig, dtype=np.int)
for a0,a1,t in conf:
    if (t!=0):
        n = int(np.log2(t))
        int_bin['baselines'][n].append((a0, a1))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((args.host, args.port))

errors = 0
packets = 0

while True:
    try:
        pkt, addr = sock.recvfrom(4120) # buffer size is 1024 bytes
        packets += 1
        t,b,o,a0,a1,x,l,data = decode_packet(pkt)
        if args.verbose and args.snap:
           print("{0:4d} {1:3d} {2:4d} {3:1d} {4:3d} {5:3d} {6:2d}".format(t, b//N_bl_per_block, b%N_bl_per_block, o, a0, a1, x), data[:8]//INTSPEC)
        elif args.verbose:
           print("{0:4d} {1:3d} {2:4d} {3:1d} {4:3d} {5:3d} {6:2d}".format(t, b//N_bl_per_block, b%N_bl_per_block, o, a0, a1, x), data[:8])

        if args.check:
           if (a0 > N_ANTS_DATA or a1 > N_ANTS_DATA): 
              print("Error! Received wrong antenna!")
           if args.snap:
              # Test data from snap
              if (a0 == 0) and (a1 == 1):
                 tspec = gen_tvg_pol(a0*2)*np.conj(gen_tvg_pol(a1*2))
                 data = data.reshape(128,4,2)//INTSPEC
                 if (a0 == a1):
                    print(a0, a1, o, np.all(tspec.real[o*128:(o+1)*128] == data[:,0,0]))
                 else:
                    print(a0, a1, o, np.all(tspec.real[o*128:(o+1)*128] == data[:,0,0]//4))
       
           else:
              n = [y for y,v in list(int_bin['baselines'].items()) if (a0,a1) in v][0]
              if not (np.all(data == int_bin['data'][n][o*1024:(o+1)*1024])):
                 print("Error!", int_bin['data'][n][:32:8], o, n, data[:32:8])
                 errors += 1
    except(KeyboardInterrupt):
       print("")
       print(("Total number of packets captured: %d"%packets))
       print(("Total number of errors: %d"%errors))
       break

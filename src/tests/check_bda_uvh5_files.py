import numpy as np
import argparse
from pyuvdata import UVData
import h5py
import argparse
from astropy.time import Time
import json
import redis

N_MAX_INTTIME = 8
N_BDABUF_BINS = 4
N_CHAN_TOTAL = 6144
N_STOKES = 4
NCHANS = 1536
N_CHAN_CATCHER = 1536
INTSPEC = 64*512*2

def get_corr_to_hera_map(nants_data=192, nants=352):
    """
    Given a redis.Redis instance, r, containing
    appropriate metadata - figure out the mapping
    of correlator index (0 - Nants_data -1) to
    hera antenna number (0 - Nants).
    """
    out_map = {} # use default values outside the range of real antennas

    r = redis.Redis('redishost')
    ant_to_snap = json.loads(r.hgetall("corr:map")['ant_to_snap'])
    for ant, pol in ant_to_snap.items():
        hera_ant_number = int(ant)
        host = pol["n"]["host"]
        chan = pol["n"]["channel"] # runs 0-5
        snap_ant_chans = r.hget("corr:snap_ants", host)
        if snap_ant_chans is None:
            continue
        corr_ant_number = json.loads(snap_ant_chans)[chan//2] #Indexes from 0-3 (ignores pol)
        out_map[corr_ant_number] = hera_ant_number

    return out_map

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


parser = argparse.ArgumentParser(description='Check contents of a uvh5 file',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('h5fname', type=str, help = 'Path to uvh5 file to test')
parser.add_argument('-c', dest='config', type=str, default='/tmp/bdaconfig.txt',
                    help='Path to BDA config file')
parser.add_argument('-p', '--paper_gpu', action='store_true', default=False,
                    help ='Test vectors obtained from hashpipe')
parser.add_argument('-f', '--fengine', action='store_true', default=False,
                    help ='Test vectors from SNAPs')
parser.add_argument('--ramp', action='store_true', default=False,
                    help='Frequency ramp in test vectors')
args = parser.parse_args()

snap_pol_map = {'e2':0,  'n0':1,
                'e6':2,  'n4':3,
                'e1':4, 'n8':5}

# Compute expected baseline pairs from
# the configuration file
conf = np.loadtxt(args.config, dtype=np.int)
bls_per_bin = np.zeros(N_BDABUF_BINS, dtype=np.int)

blpairs = []  # All pairs in config file
inttime = []
baselines = []
for ant0,ant1,t in conf:
    if (t!=0):
       baselines.append([(ant0, ant1)]*int(8//t))
       inttime.append(np.repeat(t*2, int(8//t)))
       blpairs.append((ant0, ant1))
       bls_per_bin[int(np.log2(t))] += 1
baselines = np.concatenate(baselines)
inttime   = np.asarray(np.concatenate(inttime), dtype=np.float64)

ant_1_array = np.array([x for (x,y) in baselines])
ant_2_array = np.array([y for (x,y) in baselines])


# Check integration time, ant_1_array and ant_2_array
fp = h5py.File(args.h5fname,'r')
header = fp['Header']

#print 'Integration time array...'
#assert(np.all(np.equal(header['integration_time'][:], inttime)))
#
#print 'ant_1_array..'
#assert(np.all(np.equal(header['ant_1_array'][:], ant_1_array)))
#
#print 'ant_2_array..'
#assert(np.all(np.equal(header['ant_2_array'][:], ant_2_array)))
#
#print 'Checking deltas in time array..'
#jds = Time(header['time_array'][:], format='jd')
#
#fp.close()
#
fakereal = 1
fakeimag = -2j
int_bin = {}

if (args.paper_gpu) and not (args.ramp):
   print('Checking Data for {0:s} mode with {1:s} test vectors'.format('Hashpipe','Constant'))

   # Constant
   for n in range(N_BDABUF_BINS):
      d = (2**n)*(fakereal + fakeimag)*np.ones([N_CHAN_TOTAL,N_STOKES], dtype=np.int32)
      int_bin[n] = np.sum(d.reshape(-1,4,4), axis=1)

   if (bls_per_bin[0] != 0):
      print ('Checking integration 0..')
      uv = UVData()
      uv.read(args.h5fname, run_check=False, bls=[(0,0)])
      # After catcher sum only 384/4. frequency bins come from one x-eng
      assert(np.all(np.equal(uv.data_array[0,0,:96,:], int_bin[0][:96,:])))

   if (bls_per_bin[1] != 0):
      print ('Checking one baseline in integration bin 1..')
      uv.read(args.h5fname, run_check=False, bls=[blpairs[bls_per_bin[0]]])
      assert(np.all(np.equal(uv.data_array[0, 0, :96, :], int_bin[1][:96, :])))

   if (bls_per_bin[2] != 0):
      print ('Checking one baseline in integration bin 2..')
      uv.read(args.h5fname, run_check=False, bls=[blpairs[bls_per_bin[0]+bls_per_bin[1]]])
      assert(np.all(np.equal(uv.data_array[0, 0, :96, :], int_bin[2][:96, :])))

   if (bls_per_bin[3] != 0):
      print ('Checking one baseline in integration bin 3..')
      uv.read(args.h5fname, run_check=False, bls=[blpairs[bls_per_bin[0]+bls_per_bin[1]+bls_per_bin[2]]])
      assert(np.all(np.equal(uv.data_array[0, 0, :96, :], int_bin[3][:96, :])))

if args.paper_gpu and args.ramp:
   print('Checking Data for {0:s} mode with {1:s} test vectors'.format('Hashpipe','Ramp'))

   # Ramp
   for n in range(N_BDABUF_BINS):
      x = np.reshape(np.repeat(np.arange(N_CHAN_TOTAL, dtype=np.uint8),N_STOKES),[N_CHAN_TOTAL, N_STOKES])
      int_bin[n] = np.asarray((2**n)*(fakereal + fakeimag)*x, dtype=np.complex64)

if args.fengine:
   print('Checking data for SNAPS in test vec mode')

   factor = 4

   uv = UVData()
   uv.read(args.h5fname) 

   cminfo = json.loads(fp['Header']['extra_keywords']['cminfo'].value) 
   ants = cminfo['antenna_numbers']
   out_map = get_corr_to_hera_map()

   for a0, a1 in zip(uv.ant_1_array, uv.ant_2_array):

       print(("({0:2d},{1:2d}) \t".format(a0,a1)), end=' ')

       try:
           data = uv.get_data(a1, a0)
       except (ValueError):
           print('Antenna pair (%d,%d) has no data!!!'%(a0,a1))
           continue
  

       loca0 = snap_pol_map[cminfo['correlator_inputs'][ants.index(a0)][0][:2]]
       loca1 = snap_pol_map[cminfo['correlator_inputs'][ants.index(a1)][0][:2]]
   
       tspec = gen_tvg_pol(loca0)*np.conj(gen_tvg_pol(loca1))
       #tspec = tspec[:N_CHAN_TOTAL]
       tspec = np.sum(tspec.reshape(-1,4),axis=1)[:NCHANS]

       print(np.all(np.equal(tspec, data[0,:,0]/(factor*INTSPEC))), '\t', end=' ')

       tspec = gen_tvg_pol(loca0 +1)*np.conj(gen_tvg_pol(loca1 +1))
       #tspec = tspec[:N_CHAN_TOTAL]
       tspec = np.sum(tspec.reshape(-1,4),axis=1)[:NCHANS]

       print(np.all(np.equal(tspec, data[0,:,1]/(factor*INTSPEC))))

#!/usr/bin/env python

import redis
import time
import argparse

MCNT_STEP_SIZE = 2
MCNT_XGPU_BLOCK_SIZE = 2048

def mcnts_per_second(sample_rate=500e6, spectra_len=8192):
    """
    Return number of MCNTs in 1 second. For HERA, but not in general,
    this is the same as the number of spectra in 1 second.
    sample_rate: ADC clock rate in MHz
    spectra_len: Number of frequency channels in 1 F-Engine spectra (prior to any subselecting)
    """
    return sample_rate / (spectra_len * 2)

parser = argparse.ArgumentParser(description='Turn on and off the HERA correlator',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('action',type=str,
                    help = 'Action: "start"|"stop": start|stop the correlator.')
parser.add_argument('-n', dest='acclen', type=int, default=64*512,
                    help ='Number of spectra to accumulate')
parser.add_argument('-s', dest='slices', type=int, default=2,
                    help ='Number of slices. Eg. for HERA\'s ever/odd correlator, slices=2. \
                           It is assumed that for N xhosts, hosts 1..N are slice 1, \
                           N+1..2*N are slice 2, etc.')
parser.add_argument('-t', dest='starttime', type=float, default=time.time()+5,
                    help ='UNIX time to start observations. Default is NOW+5s')
parser.add_argument('-x', dest='xhosts', type=int, default=8,
                    help ='Number of GPU hosts **per slice**')
parser.add_argument('-i', dest='xinstances_per_host', type=int, default=2,
                    help ='Number of X-engine instances per GPU host')
parser.add_argument('-r', dest='redishost', type=str, default='redishost',
                    help ='Hostname of redis server')
parser.add_argument('-S', dest='slice_by_xbox', action="store_true",
                    help ='Set to true to do even/odd samples on the same X-box')
args = parser.parse_args()

if args.action not in ['start', 'stop']:
    print('Available actions are "start" and "stop"')
    exit

# Make status keys for the various X-engine instances which identify their redis entries
status_keys = []
xinstances = args.xhosts * args.xinstances_per_host
for xhost in range(1, args.xhosts+1):
    for instance in range(args.xinstances_per_host):
        status_keys += ['hashpipe://px%d/%d/status' % (xhost, instance)]

rdb = redis.Redis(args.redishost)

if args.action == 'stop':
    rdb.publish("hashpipe:///set", 'INTSTAT=stop')

if args.action == 'start':
    # Calculate start MCNT
    mcnt_origin = int(int(rdb['corr:feng_sync_time'])/1000.)
    time_delay = args.starttime - mcnt_origin
    delay_mcnts = int(time_delay * mcnts_per_second())
    # round delay_mcnts down to an acceptable value
    round_to = MCNT_XGPU_BLOCK_SIZE * args.slices # This represents the granularity with which an integration can be started
    trig_mcnt = delay_mcnts - (delay_mcnts % round_to)
    
    trig_time = trig_mcnt / mcnts_per_second() + mcnt_origin

    #print 'Current MCNTs are:', gpumcnts
    print('Sync time is %s' % time.ctime(mcnt_origin))
    print('MCNTs per second: %.1f' % mcnts_per_second())
    print('Requested start time: %s' % time.ctime(args.starttime))
    print('Trigger MCNT: %d' % trig_mcnt)
    print('Trigger time is %.1f seconds in the future (%s)' % (trig_time - time.time(), time.ctime(trig_time)))

    # round acc_len
    acclen = int(args.acclen / MCNT_XGPU_BLOCK_SIZE) * MCNT_XGPU_BLOCK_SIZE
    print('Requested accumulation length: %d' % args.acclen)
    print('Actual accumulation length: %d' % acclen)

    # Use the hashpipe publish channel to update keys in all status buffers.
    # See the docstring at https://github.com/david-macmahon/rb-hashpipe/blob/master/bin/hashpipe_redis_gateway.rb
    # for details about formatting
    
    if args.slice_by_xbox:
        for host in range(args.slices * args.xhosts):
            hostname = 'px%d' % (host + 1)
            for slice in range(args.slices):
                msg = 'INTSYNC=%d\nINTCOUNT=%d\nINTSTAT=start\nOUTDUMPS=0' % (trig_mcnt + slice*MCNT_STEP_SIZE, acclen)
                rdb.publish('hashpipe://%s/%s/set' % (hostname, slice), msg)
                rdb.publish('hashpipe://%s/%s/set' % (hostname, slice), msg)
    else:
        for slice in range(args.slices):
            msg = 'INTSYNC=%d\nINTCOUNT=%d\nINTSTAT=start\nOUTDUMPS=0' % (trig_mcnt + slice*MCNT_STEP_SIZE, acclen)
            for host in range(args.xhosts):
                hostname = 'px%d' % (slice * args.xhosts + host + 1)
                rdb.publish('hashpipe://%s/0/set' % hostname, msg)
                rdb.publish('hashpipe://%s/1/set' % hostname, msg)

    rdb['corr:trig_mcnt'] = trig_mcnt
    rdb['corr:trig_time'] = trig_time
    rdb['corr:acc_len'] = acclen
    rdb['corr:int_time'] = acclen * args.slices * mcnts_per_second()

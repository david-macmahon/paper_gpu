#!/usr/bin/env python

import redis
import time
import argparse
import subprocess

python_source_cmd = ['source', '~/hera-venv/bin/activate']
template_cmd = ['hera_make_hdf5_template_bda.py']

def run_on_hosts(hosts, cmd, user=None, wait=True):
    if isinstance(cmd, str):
        cmd = [cmd]
    p = []
    for host in hosts:
        if user is None:
            p += [subprocess.Popen(['ssh', '%s' % (host)] + cmd)]
        else:
            p += [subprocess.Popen(['ssh', '%s@%s' % (user, host)] + cmd)]
    if wait:
        for pn in p:
            pn.wait()

parser = argparse.ArgumentParser(description='Trigger data collection on the HERA catcher node',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('host', type=str, help='Host on which to capture data')
parser.add_argument('-r', dest='redishost', type=str, default='redishost', help='Host serving redis database')
parser.add_argument('-n', dest='nfiles', type=int, default=10, help='Number of files of data to capture')
parser.add_argument('-m', dest='msperfile', type=int, default=60000, help='Number of ms of data per file')
parser.add_argument('--tag', dest='tag', type=str, default='none', help='A descriptive tag to go into data files')
parser.add_argument('-t', dest='hdf5template', type=str, default='/tmp/template.h5', help='Place to put HDF5 header template file')

args = parser.parse_args()

r = redis.Redis(args.redishost, decode_responses=True)

if len(args.tag) > 127:
  raise ValueError("Tag argument must be <127 characters!")

# Generate the meta-data template
run_on_hosts([args.host], python_source_cmd + [';'] + template_cmd + ['-c', '-r', args.hdf5template], wait=True)


#Configure runtime parameters
catcher_dict = {
  'HDF5TPLT' : args.hdf5template,
  'MSPERFIL' : args.msperfile,
  'NFILES'   : args.nfiles,
  'SYNCTIME' : r['corr:feng_sync_time'],
  'INTTIME'  : r['corr:acc_len'],
  'TAG'      : args.tag,
}
  
pubchan = 'hashpipe://%s/%d/set' % (args.host, 0)
for key, val in catcher_dict.items():
   r.publish(pubchan, '%s=%s' % (key, val))

# Only trigger after the other parameters have had ample time to write
time.sleep(0.1)
r.publish(pubchan, "TRIGGER=1")

#!/usr/bin/env python

import redis
import time
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Trigger data collection on the HERA catcher node',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('host', type=str, help='Host on which to capture data')
parser.add_argument('-r', dest='redishost', type=str, default='redishost', help='Host serving redis database')

args = parser.parse_args()

r = redis.Redis(args.redishost)

#Configure runtime parameters
catcher_dict = {
  'MSPERFIL' : 0,
  'NFILES'   : 0,
}
  
pubchan = 'hashpipe://%s/%d/set' % (args.host, 0)
for key, val in catcher_dict.items():
   r.publish(pubchan, '%s=%s' % (key, val))

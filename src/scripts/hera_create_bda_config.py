#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import h5py
import sys
import json
import logging
import numpy as np
import time
import copy
import redis

NANTS = 352


def get_cm_info():
    """Return cm_info as if from hera_mc."""
    from hera_corr_cm import redis_cm
    return redis_cm.read_cminfo_from_redis(return_as='dict')


def get_hera_to_corr_ants(r, ants=None):
    """
    Given a list of antenna numbers, get the
    corresponding correlator numbers from the
    redis database, using a redis.Redis instance (r)
    """
    ant_to_snap = json.loads(r.hgetall("corr:map")['ant_to_snap'])
    corr_nums = []
    if ants is None:
        ants = [int(a) for a in ant_to_snap.keys()]
    for a in ants:
        host = ant_to_snap['%d'%a]['n']['host']
        chan = ant_to_snap['%d'%a]['n']['channel'] # snap_input_number
        snap_ant_chans = r.hget("corr:snap_ants", host)
        if snap_ant_chans is None:
            continue
        corr_ant_number = json.loads(snap_ant_chans)[chan//2] #Indexes from 0-3 (ignores pol)
        corr_nums.append(corr_ant_number)
    return corr_nums

def create_bda_config(n_ants_data, use_cm=False, use_redis=False):
    # This does not account for BDA!!!
    if use_cm:
        cminfo = get_cm_info()
        ants = cminfo['antenna_numbers']

    if use_redis:
       r = redis.Redis('redishost')
       corr_ant_nums = get_hera_to_corr_ants(r, ants)

    else:
       corr_ant_nums = np.arange(n_ants_data)

    bl_pairs = []
    for ant0 in range(NANTS):
        for ant1 in range(ant0, NANTS, 1):
            if (ant0 in corr_ant_nums) and (ant1 in corr_ant_nums):
               bl_pairs.append([ant0, ant1, 4])
            else:
               bl_pairs.append([ant0, ant1, 0])

    return bl_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create a configuration file for BDA '\
                                     'using the correlator C+M system to get current meta-data'\
                                     'NO BDA IS CURRENTLY PERFORMED!',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('output',type=str, help = 'Path to which the configuration file should be output')
    parser.add_argument('-c', dest='use_cminfo', action='store_true', default=False,
                        help ='Use this flag to get up-to-date (hopefully) array meta-data from the C+M system')
    parser.add_argument('-r', dest='use_redis', action='store_true', default=False,
                        help ='Use this flag to get up-to-date (hopefully) f-engine meta-data from a redis server at `redishost`')
    parser.add_argument('-n', dest='n_ants_data', type=int, default=192,
                        help ='Number of antennas that have data (used if cminfo is not set)')
    args = parser.parse_args()

    baseline_pairs = create_bda_config(args.n_ants_data, use_cm = args.use_cminfo, use_redis=args.use_redis)
    np.savetxt(args.output, baseline_pairs, fmt='%d', delimiter=' ')

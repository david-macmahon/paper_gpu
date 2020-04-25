#!/usr/bin/env python

"""BDA correlator."""

import h5py
import json
import logging
import numpy as np
import time
import copy
import redis
from hera_corr_cm.handlers import add_default_log_handlers

logger = add_default_log_handlers(logging.getLogger(__file__))


def get_corr_to_hera_map(r, nants_data=192, nants=352):
    """
    Return the correlator map.

    Given a redis.Redis instance, r, containing
    appropriate metadata - figure out the mapping
    of correlator index (0 - Nants_data -1) to
    hera antenna number (0 - Nants).
    """
    out_map = np.arange(nants, nants + nants_data)  # use default values outside the range of real antennas

    # A dictionary with keys which are antenna numbers
    # of the for {<ant> :{<pol>: {'host':SNAPHOSTNAME, 'channel':INTEGER}}}
    ant_to_snap = json.loads(r.hget("corr:map", "ant_to_snap"))
    #host_to_index = r.hgetall("corr:snap_ants")
    for ant, pol in ant_to_snap.items():
        hera_ant_number = int(ant)
        host = pol["n"]["host"]
        chan = pol["n"]["channel"]  # runs 0-5
        snap_ant_chans = r.hget("corr:snap_ants", host)
        if snap_ant_chans is None:
            logger.warning("Couldn't find antenna indices for %s" % host)
            continue
        corr_ant_number = json.loads(snap_ant_chans)[chan//2] #Indexes from 0-3 (ignores pol)
        print(corr_ant_number)
        out_map[corr_ant_number] = hera_ant_number
        print("HERA antenna %d maps to correlator input %d" % (hera_ant_number, corr_ant_number))

    return out_map

def get_bl_order(n_ants):
    """
    Return the order of baseline data output by a CASPER correlator
    X engine.

    Extracted from the corr package -- https://github.com/ska-sa/corr
    """
    order1, order2 = [], []
    for i in range(n_ants):
        for j in range(int(n_ants//2),-1,-1):
            k = (i-j) % n_ants
            if i >= k: order1.append((k, i))
            else: order2.append((i, k))
    order2 = [o for o in order2 if o not in order1]
    return tuple([o for o in order1 + order2])

def get_ant_names():
    """
    Generate a list of antenna names, where position
    in the list indicates numbering in the data files.
    """
    return ["foo"]*352

def get_cm_info():
    """Return cm_info as if from hera_mc."""
    from hera_corr_cm import redis_cm
    return redis_cm.read_cminfo_from_redis(return_as='dict')

def get_antpos_enu(antpos, lat, lon, alt):
    """
    Compute the antenna positions in ENU coordinates from ECEF.

    Args:
      antpos -- array of antenna positions. Should have shape (Nants, 3).
      lat (float) -- telescope latitude, in radians
      lon (float) -- telescope longitude, in radians
      alt (float) -- telescope altitude, in meters

    Returns:
      enu -- array of antenna positions in ENU frame. Has shape (Nants, 3).
    """
    import pyuvdata.utils as uvutils
    antpos = np.asarray(antpos)
    enu  = uvutils.ENU_from_ECEF(antpos, lat, lon, alt)
    return enu

def get_antpos_ecef(antpos, lon):
    """
    Compute the antenna positions in ECEF coordinates from rotECEF

    Args:
      antpos -- array of antenna positions. Should have shape (Nants, 3).
      lon (float) -- telescope longitude, in radians

    Returns:
      ecef -- array of antenna positions in ECEF frame. Has shape (Nants, 3)
    """
    import pyuvdata.utils as uvutils
    antpos = np.asarray(antpos)
    ecef = uvutils.ECEF_from_rotECEF(antpos, lon)
    return ecef

def get_telescope_location_ecef(lat, lon, alt):
    """
    Compute the telescope location in ECEF coordinates from lat/lon/alt.

    Args:
      lat (float) -- telescope latitude, in radians
      lon (float) -- telescope longitude, in radians
      alt (float) -- telescope altitude, in meters

    Returns:
       ecef -- len(3) array of x,y,z values of telescope location in ECEF
           coordinates, in meters.
    """
    import pyuvdata.utils as uvutils
    return uvutils.XYZ_from_LatLonAlt(lat, lon, alt)


def create_header(h5, config, use_cm=False, use_redis=False):
    """
    Create an HDF5 file with appropriate datasets in a "Header"
    data group.

    inputs: h5 -- an h5py File object
            use_cm -- boolean. If True, get valid data from the hera_cm
                      system. If False, just stuff the header with fake
                      data.
    """

    INSTRUMENT = "HERA"

    #Load config file
    N_MAX_INTTIME = 8
    config = np.loadtxt(config, dtype=np.int)
    baselines = []
    integration_bin = []

    for i,t in enumerate(config[:,2]):
        if (t!=0):
           baselines.append([(config[i,0], config[i,1])]*(8//t))
           integration_bin.append(np.repeat(t, int(8//t)))
    baselines = np.concatenate(baselines)
    integration_bin = np.asarray(np.concatenate(integration_bin), dtype=np.float64)

    ant_1_array = np.array([x for (x,y) in baselines])
    ant_2_array = np.array([y for (x,y) in baselines])

    NANTS_DATA = len(set(ant_1_array))
    NANTS = 352
    NCHANS = int(2048 // 4 * 3)
    NCHANS_F = 8192
    NCHAN_SUM = 4
    ANT_DIAMETER = 14.0
    INT_TIME = 10.0
    n_bls = len(baselines) #bls = np.array(get_bl_order(NANTS_DATA))
    channel_width = 250e6 / NCHANS_F * NCHAN_SUM
    freqs = np.linspace(0, 250e6, NCHANS_F + 1)[1536 : 1536 + (8192 // 4 * 3)]
    # average over channels
    freqs = freqs.reshape(NCHANS, NCHAN_SUM).sum(axis=1) / NCHAN_SUM
    uvw = np.zeros([n_bls, 3])

    if use_cm:
        cminfo = get_cm_info()
        # add the enu co-ords
        # dict keys are bytes, not strings
        lat = cminfo[b"cofa_lat"] * np.pi / 180.0
        lon = cminfo[b"cofa_lon"] * np.pi / 180.0
        alt = cminfo[b"cofa_alt"]
        cofa_ecef = get_telescope_location_ecef(lat, lon, alt)
        antenna_positions = np.asarray(cminfo[b"antenna_positions"])
        antpos_ecef = antenna_positions + cofa_ecef
        cminfo[b"antenna_positions_enu"] = get_antpos_enu(antpos_ecef, lat, lon, alt)
    else:
        cminfo = None

    if use_redis:
        r = redis.Redis("redishost")
        fenginfo = r.hgetall("init_configuration")
        corr_to_hera_map = get_corr_to_hera_map(r, nants_data=192, nants=192)
        for n in range(baselines.shape[0]):
            baselines[n] = [corr_to_hera_map[baselines[n,0]], corr_to_hera_map[baselines[n,1]]]
        ant_1_array = np.array([x for (x,y) in baselines])
        ant_2_array = np.array([y for (x,y) in baselines])

    else:
        fenginfo = None
        # Use impossible antenna numbers to indicate they're not really valid
        corr_to_hera_map = np.arange(NANTS, NANTS+NANTS_DATA)

    header = h5.create_group("Header")
    header.create_dataset("Nants_data", dtype="<i8", data=NANTS_DATA)
    header.create_dataset("Nants_telescope", dtype="<i8", data=NANTS_DATA)
    header.create_dataset("Nbls",   dtype="<i8", data=n_bls)
    header.create_dataset("Nblts",  dtype="<i8", data=n_bls)
    header.create_dataset("Nfreqs", dtype="<i8", data=NCHANS)
    header.create_dataset("Npols",  dtype="<i8", data=4)
    header.create_dataset("Nspws",  dtype="<i8", data=1)
    # For BDA, Ntimes needs to be n_bls long
    #header.create_dataset("Ntimes", dtype="<i8", data=n_bls)
    header.create_dataset("Ntimes", dtype="<i8", data=2)
    header.create_dataset("corr_bl_order", dtype="<i8", data=np.array(baselines))
    header.create_dataset("corr_to_hera_map", dtype="<i8", data=np.array(corr_to_hera_map))
    header.create_dataset("ant_1_array_conf", dtype="<i8", data=ant_1_array)
    header.create_dataset("ant_2_array_conf", dtype="<i8", data=ant_2_array)
    header.create_dataset("antenna_diameters", dtype="<f8", data=[ANT_DIAMETER] * NANTS_DATA)
    header.create_dataset("channel_width",     dtype="<f8", data=channel_width)
    header.create_dataset("freq_array",        dtype="<f8", shape=(1, NCHANS), data=freqs) #TODO Get from config
    header.create_dataset("history",   data=np.string_("%s: Template file created\n" % time.ctime()))
    header.create_dataset("instrument", data=np.string_(INSTRUMENT))
    header.create_dataset("integration_bin", dtype="<f8", data=integration_bin)
    header.create_dataset("object_name", data=np.string_("zenith"))
    header.create_dataset("phase_type",  data=np.string_("drift"))
    header.create_dataset("polarization_array", dtype="<i8", data=[-5, -6, -7, -8])
    header.create_dataset("spw_array",      dtype="<i8", data=[0])
    header.create_dataset("telescope_name", data=np.string_("HERA"))
    header.create_dataset("vis_units",  data=np.string_("UNCALIB"))
    header.create_dataset("x_orientation", data=np.string_("NORTH"))
    if use_cm:
        # convert lat and lon from degrees -> radians
        # dict keys are bytes, not strings
        header.create_dataset("altitude",    dtype="<f8", data=cminfo[b'cofa_alt'])
        ant_pos = np.zeros([NANTS_DATA,3], dtype=np.float64)
        ant_pos_enu = np.zeros([NANTS_DATA,3], dtype=np.float64)
        ant_pos_uvw = np.zeros([NANTS,3], dtype=np.float64)
        ant_names = ["NONE"]*NANTS_DATA
        ant_nums = [-1]*NANTS_DATA
        # make uvw array
        for n, i in enumerate(cminfo[b"antenna_numbers"]):
            ant_pos_uvw[i] = cminfo[b"antenna_positions_enu"][n]
        for i,(a,b) in enumerate(baselines):
            uvw[i] = ant_pos_uvw[a] - ant_pos_uvw[b]
        # get antenna metadata only for connected antennas
        idx = 0
        for n, ant in enumerate(cminfo[b"antenna_numbers"]):
            if ant not in ant_1_array:
                continue
            ant_pos[idx]     = antenna_positions[n]
            ant_names[idx]   = np.string_(cminfo[b"antenna_names"][n])
            ant_nums[idx]    = cminfo[b"antenna_numbers"][n]
            ant_pos_enu[idx] = cminfo[b"antenna_positions_enu"][n]
            idx += 1
        # make sure we have the number we're expecting
        if idx != NANTS_DATA:
            logger.warning("Didn't get the right number of antenna positions. Expected {:d}, got {:d}".format(NANTS_DATA, idx))
        header.create_dataset("antenna_names",     dtype="|S5", shape=(NANTS_DATA,), data=ant_names)
        header.create_dataset("antenna_numbers",   dtype="<i8", shape=(NANTS_DATA,), data=ant_nums)
        header.create_dataset("antenna_positions",   dtype="<f8", shape=(NANTS_DATA,3), data=ant_pos)
        header.create_dataset("antenna_positions_enu",   dtype="<f8", shape=(NANTS_DATA,3), data=ant_pos_enu)
        header.create_dataset("latitude",    dtype="<f8", data=cminfo[b"cofa_lat"])
        header.create_dataset("longitude",   dtype="<f8", data=cminfo[b"cofa_lon"])
    else:
        header.create_dataset("altitude",    dtype="<f8", data=0.0)
        header.create_dataset("antenna_names",     dtype="|S5", shape=(NANTS,), data=["NONE"]*NANTS)
        header.create_dataset("antenna_numbers",   dtype="<i8", shape=(NANTS,), data=list(range(NANTS)))
        header.create_dataset("antenna_positions",   dtype="<f8", shape=(NANTS,3), data=np.zeros([NANTS,3]))
        header.create_dataset("antenna_positions_enu",   dtype="<f8", shape=(NANTS,3), data=np.zeros([NANTS,3]))
        header.create_dataset("latitude",    dtype="<f8", data=0.0)
        header.create_dataset("longitude",   dtype="<f8", data=0.0)

    # lst_array needs populating by receiver. Should be center of integrations in radians
    #header.create_dataset("lst_array",   dtype="<f8", data=np.zeros(n_bls))
    # time_array needs populating by receiver (should be center of integrations in JD)
    #header.create_dataset("time_array", dtype="<f8", data=np.zeros(n_bls * NTIMES))
    # uvw_needs populating by receiver: uvw = xyz(ant2) - xyz(ant1). Units, metres.
    header.create_dataset("uvw_array",  dtype="<f8", data=uvw)
    # !Some! extra_keywords need to be computed for each file
    add_extra_keywords(header, cminfo, fenginfo)


def add_extra_keywords(obj, cminfo=None, fenginfo=None):
    extras = obj.create_group("extra_keywords")
    if cminfo is not None:
        extras.create_dataset("cmver", data=np.string_(cminfo[b"cm_version"]))
        # Convert any numpy arrays to lists so they can be JSON encoded
        cminfo_copy = {}
        for key in list(cminfo.keys()):
            str_key = key.decode("utf-8")
            if isinstance(cminfo[key], np.ndarray):
                cminfo_copy[str_key] = cminfo[key].tolist()
            else:
                cminfo_copy[str_key] = cminfo[key]
        extras.create_dataset("cminfo", data=np.string_(json.dumps(cminfo_copy)))
        del(cminfo_copy)
    else:
        extras.create_dataset("cmver", data=np.string_("generated-without-cminfo"))
        extras.create_dataset("cminfo", data=np.string_("generated-without-cminfo"))
    if fenginfo is not None:
        fenginfo_copy = {}
        for key in list(fenginfo.keys()):
            str_key = key.decode("utf-8")
            if isinstance(fenginfo[key], bytes):
                fenginfo_copy[str_key] = fenginfo[key].decode("utf-8")
            else:
                fenginfo_copy[str_key] = fenginfo[key]
        extras.create_dataset("finfo", data=np.string_(json.dumps(fenginfo_copy)))
    else:
        extras.create_dataset("finfo", data=np.string_("generated-without-redis"))
    #extras.create_dataset("st_type", data=np.string_("???"))
    extras.create_dataset("duration", dtype="<f8", data=0.0)  # filled in by receiver
    extras.create_dataset("obs_id", dtype="<i8", data=0)      # "
    extras.create_dataset("startt", dtype="<f8", data=0.0)    # "
    extras.create_dataset("stopt",  dtype="<f8", data=0.0)    # "
    extras.create_dataset("corr_ver",  dtype="|S32", data=np.string_("unknown"))  # "
    extras.create_dataset("tag",  dtype="|S128", data=np.string_("unknown"))  # "

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create a template HDF5 header file, optionally '\
                                     'using the correlator C+M system to get current meta-data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('output', type=str, help = 'Path to which the template header file should be output')
    parser.add_argument('-c', dest='use_cminfo', action='store_true', default=False,
                        help ='Use this flag to get up-to-date (hopefully) array meta-data from the C+M system')
    parser.add_argument('-r', dest='use_redis', action='store_true', default=False,
                        help ='Use this flag to get up-to-date (hopefully) f-engine meta-data from a redis server at `redishost`')
    parser.add_argument('--config', type=str, default='/tmp/bdaconfig.txt',
                        help = 'BDA Configuration file to create header (taken from redis by default)')
    args = parser.parse_args()

    if args.config:
       config = args.config

    with h5py.File(args.output, "w") as h5:
        create_header(h5, config, use_cm=args.use_cminfo, use_redis=args.use_redis)

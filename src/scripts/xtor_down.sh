#!/bin/bash

show_help_and_exit() {
cat <<.
Script to totally take down the paper correlator X engines.

Usage: xtor_down.sh [-h] [SLICES]

The -h option shows this help message.

SLICES is optional.  It is a list of numbers corresponding to pxN hosts.  If
not given, it defaults to {1..8}, i.e. all pxN hosts for PSA256.

Examples:

  # Shutdown px1..px8
  $ xtor_down.sh

  # Shutdown px3 only
  $ xtor_down.sh 3

  # Shutdown px1, px2, and px3 only
  $ xtor_down.sh 1 2 3
.
exit $1
}

while getopts :h-: opt
do
  case $opt in
    h)
      show_help_and_exit 0
      ;;
    -)
      if [ $OPTARG == 'help' ]
      then
        show_help_and_exit 0
      else
        echo Invalid option: --$OPTARG
        show_help_and_exit 1
      fi
      ;;
    ?)
      echo Invalid option: -$OPTARG
      show_help_and_exit 1
      ;;
  esac
done
shift $((OPTIND-1))

# Name of PAPER server
PAPER=hera-corr-head

# Make sure we are on the PAPER server
hostname=`hostname -s`
if [ $hostname != $PAPER ]
then
  echo "This script must be run on the PAPER server (i.e. ${PAPER}), not ${hostname}"
  exit 1
fi

# If no parameters are given (the normal case), use slices {1..8}
if [ $# -eq 0 ]
then
  set {1..16}
fi

# Stop taking data
echo -n "stopping cn_rx.py"
pkill -INT cn_rx.py

# Wait upto 20 seconds for data taking to stop
timeout=20
while [ $timeout -gt 0 ] && pgrep cn_rx.py > /dev/null
do
  echo -n .
  timeout=$((timeout-1))
  sleep 1
done
echo

if [ $timeout -eq 0 ]
then
  echo 'timeout!'
fi

# Stop integrations
echo "stopping integrations"
hera_ctl.py stop

# Stop hashpipe-redis gateways
echo "stopping hashpipe-redis gateways"
redis-cli -h redishost publish hashpipe:///gateway quit

# Just to be sure
echo "killing any remaining hashpipe-redis gateways"
for x; do ssh root@px$x pkill    -f hashpipe_redis_gateway.rb; done
for x; do ssh root@px$x pkill -9 -f hashpipe_redis_gateway.rb; done

#Kill any redis loggers
echo "killing any redis loggers"
for x; do ssh root@px$x pkill    -f stdin_to_redis.py; done
for x; do ssh root@px$x pkill -9 -f stdin_to_redis.py; done

# Stop hashpipe instances
echo "killing hashpipe instances"
for x; do ssh root@px$x pkill    hashpipe; done
for x; do ssh root@px$x pkill -9 hashpipe; done

# Stop hashpipe_check_status process, which can hang if something has gone wrong previously
echo "killing hashpipe_check_status instances"
for x; do ssh root@px$x pkill    hashpipe_check_status; done
for x; do ssh root@px$x pkill -9 hashpipe_check_status; done

# Delete shared memory and semaphores
echo "deleting shared memory and semaphores"
for x; do
  for i in {0..1}; do
    ssh root@px$x hashpipe_clean_shmem -d -I $i > /dev/null
  done
done

# Just to be sure
echo "nuking any remaining shared memory and semaphores"
for x; do ssh root@px$x 'ipcs -m | awk "/0x/{print \"ipcrm -m\", \$2}" | sh'; done
for x; do ssh root@px$x 'ipcs -s | awk "/0x/{print \"ipcrm -s\", \$2}" | sh'; done
for x; do ssh root@px$x 'rm /dev/shm/*hashpipe*'; done

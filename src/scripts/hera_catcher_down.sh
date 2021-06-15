#!/bin/bash

show_help_and_exit() {
cat <<.
Script to totally take down the HERA data catcher.
.
exit $1
}

# Name of PAPER server
catcherhost=hera-sn1

# Stop hashpipe-redis gateways
echo "stopping hashpipe-redis gateways"
redis-cli -h redishost publish hashpipe://$catcherhost/gateway quit

# Just to be sure
echo "killing any remaining hashpipe-redis gateways"
ssh root@$catcherhost pkill -f hashpipe_redis_gateway.rb
ssh root@$catcherhost pkill -f -9 hashpipe_redis_gateway.rb

#Kill any redis loggers
echo "killing any redis loggers"
ssh root@$catcherhost pkill    -f stdin_to_redis.py
ssh root@$catcherhost pkill -9 -f stdin_to_redis.py

# Stop hashpipe instances
echo "killing hashpipe instances"
ssh root@$catcherhost pkill -f hashpipe
ssh root@$catcherhost pkill -f -9 hashpipe

# Delete shared memory and semaphores
echo "deleting shared memory and semaphores"
ssh root@$catcherhost hashpipe_clean_shmem -d -I 0 > /dev/null

# Just to be sure
echo "nuking any remaining shared memory and semaphores"
ssh root@$catcherhost 'ipcs -m | awk "/0x/{print \"ipcrm -m\", \$2}" | sh'
ssh root@$catcherhost 'ipcs -s | awk "/0x/{print \"ipcrm -s\", \$2}" | sh'
ssh root@$catcherhost 'rm /dev/shm/*hashpipe*'

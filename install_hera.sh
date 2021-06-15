#! /bin/bash

#default to 192 antennas
n_ants=192

while getopts ":a:h:" opt; do
  case ${opt} in
    a )
      n_ants=$OPTARG
      ;;
    \? )
      echo "Usage: $(basename $0) [-a]"
      echo "  -a : number of antennas to compile for and install"
      exit
      ;;
  esac
done

echo "Building and installing for ${n_ants} antennas"
sleep 1;

cd xGPU/src

make clean
make NSTATION=$n_ants NFREQUENCY=384 NTIME=2048 NTIME_PIPE=1024 CUDA_ARCH=sm_61 DP4A=yes
sudo make install prefix="/usr/local/xgpu-1080-dp4a-384c-${n_ants}a"

cd ../../src

./configure --with-xgpu="/usr/local/xgpu-1080-dp4a-384c-${n_ants}a"
make clean
make
sudo make install

#! /bin/bash

ethtool -S eth3 | grep rx_dropped
ethtool -S eth3 | grep vport_dropped
ethtool -S eth3 | grep rx_fifo_errors
ethtool -S eth3 | grep rx_missed_errors
ethtool -S eth5 | grep rx_dropped
ethtool -S eth5 | grep vport_dropped
ethtool -S eth5 | grep rx_fifo_errors
ethtool -S eth5 | grep rx_missed_errors

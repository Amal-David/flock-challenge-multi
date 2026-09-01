#!/bin/sh
export FLOCK_GAP_TIMING=1
export FLOCK_COMMIT_TIMING=1
export FLOCK_ZC_TIMING=1
export FLOCK_OPEN_TIMING=1
export LIG_PROVE_TRACE=1
export LINCHECK_TRACE=1
export PCS_TRACE=1
export PERM_TRACE=1
export MERKLE_TRACE=1
exec /home/ubuntu/stfmp-control-worker "$@" 2>>/home/ubuntu/cff-profile-trusted.log

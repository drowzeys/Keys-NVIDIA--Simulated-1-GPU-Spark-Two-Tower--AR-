#!/bin/bash
while :; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$a" -lt 3000000 ]; then
    docker rm -f twotower-ar 2>/dev/null
    echo "$(date '+%F %T') FASTKILL fired at ${a}KB avail" >> /home/keyspark/oom-fastkill.log
    sleep 10
  fi
  sleep 2
done

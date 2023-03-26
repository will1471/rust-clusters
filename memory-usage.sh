#!/bin/bash

set -e
#set -x

for alg in "cluster-ndarray " "cluster-ndarray2" "cluster-ndarray3" "cluster-ndarray4"; do
  for size in "1k" "5k" "10k" "20k" "30k" "40k"; do
    bytes=$(cargo run --release --features dhat-heap -- $alg $size.json /dev/null 2>&1 | grep -o "t-gmax: [0-9,]\+ bytes" | sed 's/[^0-9]*//g')
    megabytes=$(qalc "$bytes bytes to MB" | grep -o "[0-9.]\+ ....bytes")
    echo "$alg - $size vectors = $megabytes";
  done
done

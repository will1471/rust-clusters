#!/bin/bash

set -e
#set -x

for alg in "cluster-ndarray " "cluster-ndarray2"; do
  for size in " 1k" " 5k" "10k" "20k" "30k" "40k"; do
    run1=$(cargo run --release -- $alg $size.json /dev/null 2>&1 | grep Finished\ main_cluster | grep -o 'PT[0-9\.]\+S' | sed 's/[^0-9\.]//g')
    run2=$(cargo run --release -- $alg $size.json /dev/null 2>&1 | grep Finished\ main_cluster | grep -o 'PT[0-9\.]\+S' | sed 's/[^0-9\.]//g')
    run3=$(cargo run --release -- $alg $size.json /dev/null 2>&1 | grep Finished\ main_cluster | grep -o 'PT[0-9\.]\+S' | sed 's/[^0-9\.]//g')
    avg=$(qalc --set "prec 4" "($run1 + $run2 + $run3) / 3" | grep -o '[0-9\.]\+$')
    echo "$alg - $size = $avg seconds avg 3 runs ($run1 $run2 $run3)";
  done
done



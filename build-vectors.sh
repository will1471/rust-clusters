#!/bin/bash

set -x
set -e

DATASET="data.csv"

csvtool col 6 $DATASET | head -n 1000 > 1k.txt
csvtool col 6 $DATASET | head -n 5000 > 5k.txt
csvtool col 6 $DATASET | head -n 10000 > 10k.txt
csvtool col 6 $DATASET | head -n 20000 > 20k.txt
csvtool col 6 $DATASET | head -n 30000 > 30k.txt
csvtool col 6 $DATASET | head -n 40000 > 40k.txt

cargo run --release -- vectors 1k.txt 1k.json
cargo run --release -- vectors 5k.txt 5k.json
cargo run --release -- vectors 10k.txt 10k.json
cargo run --release -- vectors 20k.txt 20k.json
cargo run --release -- vectors 30k.txt 30k.json
cargo run --release -- vectors 40k.txt 40k.json

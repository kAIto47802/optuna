#!/bin/bash

branch=$(git rev-parse --abbrev-ref HEAD | tr '/' '_')
mkdir -p results
outfile="results/apply_logs_${branch}.csv"
echo "run,log_number,memory" > $outfile
for run in {1..5}; do
    python read_logs.py | sed "s/^/$run,/" >> "$outfile"
done


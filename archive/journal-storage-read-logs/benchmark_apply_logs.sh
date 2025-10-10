#!/bin/bash

branch=$(git rev-parse --abbrev-ref HEAD | tr '/' '_')
mkdir -p results
outfile="results/apply_logs_${branch}.csv"
echo "length,run,before,after" > $outfile
for run in {1..5}; do
    for k in {1..19}; do
        length=$((1<<k))
        result=$(python read_logs.py --log_length $length)
        echo "$length,$run,$result" >> $outfile
    done
done


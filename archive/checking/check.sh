#!/bin/bash

git switch master
master=$(git rev-parse HEAD)
python check.py > ${master}.txt

git switch $1
refactor=$(git rev-parse HEAD)
python check.py > ${refactor}.txt

if cmp -s ${master}.txt ${refactor}.txt; then
    echo "Both branches produce the same output."
    exit 0
else
    echo "The outputs differ between the branches."
    exit 1
fi
#!/bin/bash


function_ids=(2 3 16 17 20 22)
dimensions=(5 20)
versions=('v1.14' 'v1.15')


for function_id in "${function_ids[@]}"; do
  for dimension in "${dimensions[@]}"; do
    for version in "${versions[@]}"; do
      if [[ "$function_id" -eq 3 && "$dimension" -eq 5 ]]; then
        echo "Skipping function_id 2 with dimension 5"
        continue
      fi
      source "scipy-${version}/bin/activate"
      echo "Running function $function_id with dimension $dimension in SciPy version $version"
      python benchmark_scipy2.py --function_id "$function_id" --dimension "$dimension"
    done
  done
done

#!/bin/bash


function_ids=(2 3 16 17 20 22)
dimensions=(5 20)
versions=('v1.14' 'v1.15')


for function_id in "${function_ids[@]}"; do
  for dimension in "${dimensions[@]}"; do
    for version in "${versions[@]}"; do
      python plot_scipy2.py \
        --commits \
          2bfc0bb5290c7ab44a0bcdbfe7aea4b7626307e5 \
          a6487ff321891c0936e8bb5e18d0dd710f2e08f4 \
        --comments "This PR" "Original" \
        --function_id "$function_id" \
        --dimension "$dimension"
    done
  done
done


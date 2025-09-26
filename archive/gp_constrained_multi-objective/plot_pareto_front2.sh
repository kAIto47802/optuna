#!/bin/bash

samplers=("gp" "gp_wo_constraints" "tpe" "nsgaii")

for sampler in "${samplers[@]}"; do
    python plot_pareto_front.py --constraint_type 2 --function_id 2 --sampler $sampler --file_format png
done

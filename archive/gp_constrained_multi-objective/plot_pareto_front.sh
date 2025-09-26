#!/bin/bash

samplers=("gp" "gp_wo_constraints" "tpe" "nsgaii")

for sampler in "${samplers[@]}"; do
    python plot_pareto_front.py --constraint_type 2 --function_id 2 --sampler $sampler
done

pdflatex place_figure.tex

convert -density 600 place_figure.pdf -quality 300 -background white -alpha remove -alpha off results/parato_front.png
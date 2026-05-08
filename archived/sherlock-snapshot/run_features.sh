#!/bin/bash
do
for y in 2019 2021
do
    for m in `seq -w 1 12`
    do
	echo python3 features.py $y$m
	python features.py $y$m
    done
done
done
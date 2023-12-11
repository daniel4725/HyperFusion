#!/bin/bash

part_len=100
for part in $(seq 0 1 10); do
args="--part_num $part --part_len $part_len"
echo $args
/home/galialab/miniconda2/envs/pipe_37/bin/python /media/data2/tempDaniel/main.py $args
done

exit 0
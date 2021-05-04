#!/bin/sh
### General Options
#BSUB -J KClosestNeighbours
#BSUB -q hpc
#BSUB -W 16:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUP -M 25GB
#BSUB -u s174169@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o Output_%J.out

echo "Running job"
python calcclosest.py
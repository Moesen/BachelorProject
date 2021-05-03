#!/bin/sh
### General Options
#BSUB -J KClosestNeighbours
#BSUB -q kcn
#BSUP -W 1:00
#BSUP -n 1
#BSUP -R "span[hosts=1]"
#BSUP -R "rusage[mem=20GB]"
#BSUP -M 25GB
#BSUP -u s174169@student.dtu.dk
#BSUP -B
#BSUP -N
#BSUP -o Output_%J.out
#BSUP -e Error_%J.err

echo "Running job"
python graphprediction.py
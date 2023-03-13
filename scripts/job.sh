#!/bin/sh

##Job Script for FYP

#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --job-name=deepface
#SBATCH --output=./output/output_%x_%j.out
#SBATCH --error=./output/error_%x_%j.err

CUDA_VISIBLE_DEVICES=0 python job.py
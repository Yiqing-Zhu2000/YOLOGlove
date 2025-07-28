#!/bin/bash
#SBATCH --account=def-skrishna
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:30
#SBATCH --output=yqDemo1-%j.out
#SBATCH --mail-user=yiqing.zhu2@mail.mcgill.ca
#SBATCH --mail-type=BEGIN,END,FAIL


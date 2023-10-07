#!/bin/bash
#PBS -N CleanerCCBS256val
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8,gpus=1,mem=15gb,walltime=24:00:00
#PBS -M sharmaa@informatik.uni-freiburg.de
#PBS -j oe
source ~/.bashrc
conda activate ngroupvit
cd /misc/student/sharmaa/groupvit/GroupViT 
./tools/dist_launch.sh new_run_gvit.py configs/upper_limit_dataloader_group_vit_gcc_yfcc_30e.yml 1 --resume /misc/lmbraid21/sharmaa/checkpoints/group_vit_gcc_yfcc_30e-879422e0.pth --opts train.finetune.only_grouping=true --output /misc/lmbraid21/sharmaa/outputs/gs2_bf_cc256_val
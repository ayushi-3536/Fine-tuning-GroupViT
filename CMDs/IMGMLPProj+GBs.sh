#!/bin/bash
#PBS -N IMGMLP+GBs_0.0001
#PBS -S /bin/bash
#PBS -l hostlist=^track,nodes=1:ppn=16,gpus=2,mem=36gb,walltime=24:00:00
#PBS -M sharmaa@informatik.uni-freiburg.de
#PBS -j oe
source ~/.bashrc
conda activate ngroupvit
cd /misc/student/sharmaa/groupvit/GroupViT 
./tools/dist_launch.sh main_group_vit.py configs/entropy_group_vit_gcc_yfcc_30e.yml 2 --resume /misc/lmbraid21/sharmaa/checkpoints/group_vit_gcc_yfcc_30e-879422e0.pth --opts train.finetune.only_grouping=true  --output /misc/lmbraid21/sharmaa/outputs/gs2_bothgb_lr0.01

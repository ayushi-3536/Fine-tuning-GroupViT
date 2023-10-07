#!/bin/bash
#PBS -N 2ndGB+MLP
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8,gpus=1,mem=15gb,walltime=24:00:00
#PBS -M sharmaa@informatik.uni-freiburg.de
#PBS -j oe
source ~/.bashrc
conda activate ngroupvit
cd /misc/student/sharmaa/groupvit/GroupViT 
./tools/dist_launch.sh main_group_vit.py configs/entropy_group_vit_gcc_yfcc_30e.yml 1 --resume /misc/lmbraid21/sharmaa/checkpoints/group_vit_gcc_yfcc_30e-879422e0.pth --opts train.finetune.only_grouping2=true train.finetune.only_img_projector=true --output /misc/lmbraid21/sharmaa/outputs/gs2_gb2_imgproj_new
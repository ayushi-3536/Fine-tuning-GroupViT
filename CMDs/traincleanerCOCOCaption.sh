#!/bin/bash
#PBS -N UpperBoundCCTrain
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=16,gpus=2,mem=40gb,walltime=24:00:00
#PBS -M sharmaa@informatik.uni-freiburg.de
#PBS -j oe
source ~/.bashrc
conda activate ngroupvit
cd /misc/student/sharmaa/groupvit/GroupViT 
./tools/dist_launch.sh new_run_gvit.py configs/train_upper_limit_dataloader_group_vit_gcc_yfcc_30e.yml 2 --resume /misc/lmbraid21/sharmaa/checkpoints/group_vit_gcc_yfcc_30e-879422e0.pth --opts train.finetune.only_grouping=true --output /misc/lmbraid21/sharmaa/outputs/gs2_newupperlimit_train_just_caption
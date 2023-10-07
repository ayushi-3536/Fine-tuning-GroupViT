#!/bin/bash
#PBS -N HR_384
#PBS -S /bin/bash
#PBS -l hostlist=^chip,nodes=1:ppn=16,gpus=2,mem=40gb,walltime=24:00:00
#PBS -M sharmaa@informatik.uni-freiburg.de
#PBS -j oe
source ~/.bashrc
conda activate ngroupvit
cd /misc/student/sharmaa/groupvit/GroupViT 
./tools/dist_launch.sh main_group_vit.py configs/entropy_group_vit_gcc_yfcc_30e.yml 2 --resume /misc/lmbraid21/sharmaa/checkpoints/group_vit_gcc_yfcc_30e-879422e0.pth --opts train.finetune.only_grouping=true model.img_encoder.img_size=384 data.img_aug.img_size=384 --output /misc/lmbraid21/sharmaa/outputs/gs2_newGR384_new